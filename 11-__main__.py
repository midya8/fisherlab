import json
import sys
from pathlib import Path
from . import audit, encode, run_demo


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv == ['demo']:
            out = run_demo()
        elif len(argv) == 2 and argv[0] == 'audit':
            raw = Path(argv[1]).read_bytes()
            if len(raw) > 100000:
                raise ValueError('input exceeds100KB')
            def unique(pairs):
                result = {}
                for k, v in pairs:
                    if k in result:
                        raise ValueError('duplicate JSON field')
                    result[k] = v
                return result
            obj = json.loads(raw, object_pairs_hook=unique)
            if (not isinstance(obj, dict) or set(obj) != {'version', 'probabilities', 'rewards', 'coordinates', 'damping'}
                    or type(obj['version']) is not int or obj['version'] != 1):
                raise ValueError('expected version1 and exactly probabilities/rewards/coordinates/damping')
            out = audit(obj['probabilities'], obj['rewards'], obj['coordinates'], obj['damping'])
        else:
            raise ValueError('usage: python -m fisherlab demo | audit INPUT.json')
        print(json.dumps(encode(out), indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, OverflowError, RecursionError) as e:
        print('fisherlab: '+str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
