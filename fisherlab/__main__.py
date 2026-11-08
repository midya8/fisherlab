import json
import sys
from pathlib import Path
from . import audit, encode, run_demo, audit_contexts, run_context_demo


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv == ['demo']:
            out = run_demo()
        elif argv == ['context-demo']:
            out = run_context_demo()
        elif len(argv) == 2 and argv[0] in ['audit', 'contexts']:
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
            fields = ({'version', 'probabilities', 'rewards', 'coordinates', 'damping'} if argv[0]=='audit'
                      else {'version','weights','contexts','damping'})
            if (not isinstance(obj, dict) or set(obj) != fields
                    or type(obj['version']) is not int or obj['version'] != 1):
                raise ValueError('expected integer version1 and exact fields: '+','.join(sorted(fields)))
            out = (audit(obj['probabilities'], obj['rewards'], obj['coordinates'], obj['damping']) if argv[0]=='audit'
                   else audit_contexts(obj['weights'], obj['contexts'], obj['damping']))
        else:
            raise ValueError('usage: python -m fisherlab demo | audit INPUT.json | context-demo | contexts INPUT.json')
        print(json.dumps(encode(out), indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, OverflowError, RecursionError) as e:
        print('fisherlab: '+str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
