# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class IIDAutoLRAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "auto lr training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'fd_env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')


class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None


class IIDTopAccThenMinECESelectionMethod(SelectionMethod):
    """Among checkpoints with top-pct validation accuracy, pick min validation ECE.

    - Validation accuracy/ECE are computed over training domains' out-splits
      (i.e., mean of env{i}_out_* for i != test_env).
    - Reports test metric as env{test}_in_acc to match other methods.

    Control the percentile threshold via class attribute TOP_PCT (0-100).
    """
    name = "training-domain: top-pct-acc then min-ECE"
    TOP_PCT = 20.0  # percent of top val-acc checkpoints to consider

    @classmethod
    def _step_metrics(self, record):
        test_env = record['args']['test_envs'][0]
        # Collect val (train envs) out-acc and out-ece across envs != test_env
        val_out_accs = []
        val_out_eces = []
        for i in itertools.count():
            acc_key = f'env{i}_out_acc'
            if acc_key not in record:
                break
            if i == test_env:
                continue
            val_out_accs.append(record[acc_key])
            ece_key = f'env{i}_out_ece'
            # Missing ECE should exclude this env from ECE averaging
            if ece_key in record:
                val_out_eces.append(record[ece_key])

        if not val_out_accs:
            return None

        val_acc = float(np.mean(val_out_accs))

        # If ECE not present, skip this record for our method
        if not val_out_eces:
            return None
        val_ece = float(np.mean(val_out_eces))

        test_in_acc_key = f'env{test_env}_in_acc'
        test_acc = record[test_in_acc_key]
        return {
            'val_acc': val_acc,
            'val_ece': val_ece,
            'test_acc': test_acc,
            'step': record['step']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None

        # Build per-step metrics
        metrics = []
        for r in test_records:
            m = self._step_metrics(r)
            if m is not None:
                metrics.append(m)

        if not metrics:
            # Fallback: if no ECE available, behave like IIDAccuracy
            return IIDAccuracySelectionMethod.run_acc(run_records)

        # Filter to top-pct by val_acc
        top_pct = float(self.TOP_PCT)
        top_pct = min(max(top_pct, 0.0), 100.0)
        k = max(1, int(np.ceil((top_pct / 100.0) * len(metrics))))
        metrics_sorted_by_acc = sorted(metrics, key=lambda d: d['val_acc'], reverse=True)
        top_bucket = metrics_sorted_by_acc[:k]

        # Select by minimum val ECE; tie-break by higher val_acc, then later step
        top_bucket_sorted = sorted(top_bucket, key=lambda d: (d['val_ece'], -d['val_acc'], d['step']))
        chosen = top_bucket_sorted[0]
        return {
            'val_acc': chosen['val_acc'],
            'test_acc': chosen['test_acc']
        }


class _TopAccThenSignalBase(SelectionMethod):
    """Base: Among top-pct val-acc, select by a scalar signal.

    Subclasses must define:
      - NAME: human-readable method name
      - SIGNAL_KEY: one of 'sic', 'sie', 'sikl'
      - DIRECTION: 'min' or 'max' (minimize or maximize the signal)
      - TOP_PCT: float percentile
    """
    NAME = ""
    SIGNAL_KEY = ""
    DIRECTION = "min"
    TOP_PCT = 20.0

    @classmethod
    def _step_metrics(self, record):
        test_env = record['args']['test_envs'][0]
        # val acc
        val_out_accs = []
        # signal over training envs' out splits
        val_out_sigs = []
        for i in itertools.count():
            acc_key = f'env{i}_out_acc'
            if acc_key not in record:
                break
            if i == test_env:
                continue
            val_out_accs.append(record[acc_key])
            sig_key = f'env{i}_out_{self.SIGNAL_KEY}'
            if sig_key in record:
                val_out_sigs.append(record[sig_key])

        if not val_out_accs or not val_out_sigs:
            return None

        val_acc = float(np.mean(val_out_accs))
        val_sig = float(np.mean(val_out_sigs))
        test_in_acc_key = f'env{test_env}_in_acc'
        return {
            'val_acc': val_acc,
            'val_sig': val_sig,
            'test_acc': record[test_in_acc_key],
            'step': record['step']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        metrics = []
        for r in test_records:
            m = self._step_metrics(r)
            if m is not None:
                metrics.append(m)
        if not metrics:
            return IIDAccuracySelectionMethod.run_acc(run_records)

        top_pct = float(self.TOP_PCT)
        top_pct = min(max(top_pct, 0.0), 100.0)
        k = max(1, int(np.ceil((top_pct / 100.0) * len(metrics))))
        metrics_sorted_by_acc = sorted(metrics, key=lambda d: d['val_acc'], reverse=True)
        top_bucket = metrics_sorted_by_acc[:k]

        if self.DIRECTION == 'max':
            best = sorted(top_bucket, key=lambda d: (-d['val_sig'], -d['val_acc'], d['step']))[0]
        else:
            best = sorted(top_bucket, key=lambda d: (d['val_sig'], -d['val_acc'], d['step']))[0]
        return {
            'val_acc': best['val_acc'],
            'test_acc': best['test_acc']
        }


class IIDTopAccThenMinSICSelectionMethod(_TopAccThenSignalBase):
    name = "training-domain: top-pct-acc then min-SIC"
    SIGNAL_KEY = 'sic'
    DIRECTION = 'min'


class IIDTopAccThenMaxSIESelectionMethod(_TopAccThenSignalBase):
    name = "training-domain: top-pct-acc then max-SIE"
    SIGNAL_KEY = 'sie'
    DIRECTION = 'max'


class IIDTopAccThenMinSIKLSelectionMethod(_TopAccThenSignalBase):
    name = "training-domain: top-pct-acc then min-SIKL"
    SIGNAL_KEY = 'sikl'
    DIRECTION = 'min'
