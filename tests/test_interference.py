import copy
from fractions import Fraction as Q
import random
import unittest
from fisherlab.interference import audit_interference


class InterferenceTests(unittest.TestCase):
    def fixture(self):
        return [dict(probabilities=['1/2','1/2'],rewards=[0,r],coordinates=[[0],[1]]) for r in [1,-2]]

    def test_closed_form_shared_tradeoff(self):
        x=audit_interference(['3/4','1/4'],self.fixture())
        self.assertEqual(x['interaction_matrix'],[[Q(1,4),-Q(1,2)],[-Q(1,2),Q(1)]])
        self.assertEqual(x['context_reward_derivatives'],[Q(1,16),-Q(1,8)])
        self.assertEqual(x['objective_derivative'],Q(1,64))
        self.assertEqual([s['solution'] for s in x['context_gradient_systems']],[[1],[-2]])

    def test_binary_closed_forms(self):
        rng=random.Random(893)
        for _ in range(20):
            counts=[rng.randint(1,7) for _ in range(3)];weights=[Q(n,sum(counts)) for n in counts]
            ps=[Q(rng.randint(1,7),8) for _ in counts];slopes=[rng.randint(-3,3) for _ in counts];rewards=[rng.randint(-4,4) for _ in counts]
            contexts=[dict(probabilities=[1-p,p],rewards=[0,r],coordinates=[[0],[a]]) for p,a,r in zip(ps,slopes,rewards)]
            metric=sum(w*p*(1-p)*a*a for w,p,a in zip(weights,ps,slopes))
            gs=[p*(1-p)*a*r for p,a,r in zip(ps,slopes,rewards)]
            expected=[[g*h/metric if metric else Q(0) for h in gs] for g in gs]
            self.assertEqual(audit_interference(weights,contexts)['interaction_matrix'],expected)

    def test_independent_parameters_and_singular_gauges(self):
        c=self.fixture();c[0]['coordinates']=[[0,0,7],[1,0,7]];c[1]['coordinates']=[[0,0,-2],[0,1,-2]]
        x=audit_interference(['3/4','1/4'],c)
        self.assertEqual(x['interaction_matrix'],[[Q(1,3),0],[0,4]])
        self.assertEqual(x['context_reward_derivatives'],[Q(1,4),1])
        self.assertEqual([s['rank'] for s in x['context_gradient_systems']],[2,2])
        self.assertTrue(all(not any(r) for r in x['null_gradient_residuals']))

    def test_reparameterization_and_reward_transforms(self):
        c=[dict(probabilities=['1/2','1/3','1/6'],rewards=[0,1,3],coordinates=[[1,0],[0,1],[0,0]]),dict(probabilities=['1/4','1/4','1/2'],rewards=[2,-1,4],coordinates=[[0,1],[1,0],[0,0]])]
        w=['2/3','1/3'];x=audit_interference(w,c)
        transformed=copy.deepcopy(c)
        for row in transformed:
            row['coordinates']=[[a+2*b,3*a+7*b,a+2*b] for a,b in row['coordinates']]
        self.assertEqual(audit_interference(w,transformed)['interaction_matrix'],x['interaction_matrix'])
        altered=copy.deepcopy(c)
        for row in altered:row['rewards']=[-2*r+5 for r in row['rewards']]
        self.assertEqual(audit_interference(w,altered)['interaction_matrix'],[[4*h for h in row] for row in x['interaction_matrix']])
        for v in ([1,0],[0,1],[1,1],[1,-1],[-2,3]):
            self.assertGreaterEqual(sum(Q(a)*h*Q(b) for a,row in zip(v,x['interaction_matrix']) for h,b in zip(row,v)),0)

    def test_order_splitting_frozen_and_isolation(self):
        c=self.fixture();old=copy.deepcopy(c);state=random.getstate();x=audit_interference(['3/4','1/4'],c)
        rev=audit_interference(['1/4','3/4'],list(reversed(c)))
        self.assertEqual(rev['interaction_matrix'],[list(reversed(r)) for r in reversed(x['interaction_matrix'])])
        split=audit_interference(['1/4','1/2','1/4'],[c[0],c[0],c[1]])
        self.assertEqual(split['context_reward_derivatives'],[x['context_reward_derivatives'][0]]*2+[x['context_reward_derivatives'][1]])
        frozen=copy.deepcopy(c)
        for row in frozen:row['coordinates']=[[3],[3]]
        self.assertEqual(audit_interference(['3/4','1/4'],frozen)['interaction_matrix'],[[0,0],[0,0]])
        self.assertEqual(c,old);self.assertEqual(random.getstate(),state)
        x['context_audit']['local_audits'][0]['coordinates'][0][0]=99
        self.assertEqual(c,old)

    def test_validation_delegates_without_relaxing_contract(self):
        for w in ([True,False],[0,1],['1/2','1/3']):
            with self.assertRaises(ValueError):audit_interference(w,self.fixture())
        with self.assertRaises(ValueError):audit_interference(['3/4','1/4'],self.fixture(),damping=0)

    def test_independent_determinant_reference(self):
        # Compute reward gradients and metric directly from centered raw charts,
        # then use a separate determinant/Cramer solve, never production helpers.
        import itertools
        def det(m):
            n=len(m);total=Q(0)
            for perm in itertools.permutations(range(n)):
                sign=(-1)**sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n))
                term=Q(sign)
                for i,j in enumerate(perm):term*=m[i][j]
                total+=term
            return total
        rng=random.Random(271)
        for d in [2,3]:
            for _ in range(8):
                contexts=[]
                for k in [2,3,4,2,3,4]:
                    counts=[rng.randint(1,5) for _ in range(k)]
                    contexts.append(dict(probabilities=[Q(n,sum(counts)) for n in counts],rewards=[rng.randint(-3,3) for _ in counts],coordinates=[[rng.randint(-3,3) for _ in range(d)] for _ in counts]))
                w=[Q(1,6)]*6;metric=[[Q(0) for _ in range(d)] for _ in range(d)];gs=[]
                for weight,c in zip(w,contexts):
                    p=c['probabilities'];r=c['rewards'];a=c['coordinates'];means=[sum(pi*row[j] for pi,row in zip(p,a)) for j in range(d)];scores=[[x-m for x,m in zip(row,means)] for row in a]
                    gs.append([sum(pi*si[j]*ri for pi,si,ri in zip(p,scores,r)) for j in range(d)])
                    for j in range(d):
                        for h in range(d):metric[j][h]+=weight*sum(pi*si[j]*si[h] for pi,si in zip(p,scores))
                denominator=det(metric);self.assertNotEqual(denominator,0);solutions=[]
                for g in gs:
                    solution=[]
                    for j in range(d):
                        replaced=[list(row) for row in metric]
                        for i in range(d):replaced[i][j]=g[i]
                        solution.append(det(replaced)/denominator)
                    solutions.append(solution)
                expected=[[sum(a*b for a,b in zip(g,v)) for v in solutions] for g in gs]
                x=audit_interference(w,contexts);self.assertEqual(x['interaction_matrix'],expected)
                self.assertEqual([s['solution'] for s in x['context_gradient_systems']],solutions)

    def test_strict_cli_and_api_export(self):
        import json,subprocess,sys,tempfile
        from pathlib import Path
        from fisherlab import audit_interference as exported,encode
        self.assertIs(exported,audit_interference)
        payload=dict(version=1,weights=['3/4','1/4'],contexts=self.fixture(),damping='1/10')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.json';path.write_text(json.dumps(payload))
            result=subprocess.run([sys.executable,'-m','fisherlab','interference',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stderr,'')
            self.assertEqual(json.loads(result.stdout),encode(audit_interference(payload['weights'],payload['contexts'])))
            for text in [json.dumps(dict(payload,version=True)),json.dumps(dict(payload,extra=1)),json.dumps({k:v for k,v in payload.items() if k!='damping'}),'{"version":1,"version":1}', '['*1500, 'x'*100001]:
                path.write_text(text);result=subprocess.run([sys.executable,'-m','fisherlab','interference',str(path)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,'');self.assertTrue(result.stderr.startswith('fisherlab:'));self.assertNotIn('Traceback',result.stderr)
