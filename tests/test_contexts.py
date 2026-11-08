import copy
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from fisherlab import audit, audit_contexts, encode, run_context_demo


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.contexts=[{'probabilities':['1/2','1/2'],'rewards':[0,1],'coordinates':[[0],[1]]},
                       {'probabilities':['1/2','1/2'],'rewards':[0,-2],'coordinates':[[0],[1]]}]
        self.weights=['3/4','1/4']

    def test_exact_shared_tradeoff(self):
        x=audit_contexts(self.weights,self.contexts);n=x['natural']
        self.assertEqual(x['pullback_fisher'],[[Q(1,4)]])
        self.assertEqual(x['parameter_gradient'],[Q(1,16)])
        self.assertEqual(n['parameter_direction'],[Q(1,4)])
        self.assertEqual(n['objective_derivative'],Q(1,64))
        self.assertEqual(n['contexts'][0]['reward_derivative'],Q(1,16))
        self.assertEqual(n['contexts'][1]['reward_derivative'],-Q(1,8))
        self.assertEqual(n['residual_length_squared'],Q(27,64))
        self.assertEqual(n['probability_length_squared'],Q(1,64))
        self.assertEqual(x['full_conditional_target_length_squared'],Q(7,16))
        self.assertEqual(n['pythagorean_residual'],0)
        self.assertEqual(n['projection_inner_product'],0)
        self.assertFalse(x['full_conditional_reachable'])

    def test_independent_contexts_full_reference(self):
        x=run_context_demo()['weighted_independent'];n=x['natural']
        self.assertEqual(x['pullback_fisher'],[[Q(3,16),0],[0,Q(1,16)]])
        self.assertEqual(x['parameter_gradient'],[Q(3,16),-Q(1,8)])
        self.assertEqual(n['parameter_direction'],[1,-2])
        self.assertEqual(n['objective_derivative'],Q(7,16))
        self.assertEqual(n['residual_length_squared'],0)
        self.assertTrue(x['full_conditional_reachable'])
        self.assertEqual([c['probability_tangent'] for c in n['contexts']],[[-Q(1,4),Q(1,4)],[Q(1,2),-Q(1,2)]])

    def test_uniform_objective_not_silent_reweighting(self):
        x=run_context_demo();u=x['uniform_shared_different_objective']
        self.assertEqual(u['natural']['parameter_direction'],[-Q(1,2)])
        self.assertEqual(u['natural']['objective_derivative'],Q(1,16))
        self.assertEqual(x['uniform_direction_evaluated_on_original_weighted_objective'],-Q(1,32))

    def test_flattened_joint_softmax_is_different_model(self):
        x=run_context_demo();frozen=x['context_gauge_conditional_frozen'];naive=x['naive_flattened_joint_softmax_different_model']
        self.assertEqual(frozen['pullback_fisher'],[[0]])
        self.assertEqual(frozen['parameter_gradient'],[0])
        self.assertEqual(frozen['system']['null_basis'],[[1]])
        self.assertEqual(frozen['null_probability_tangents'],[[[0,0],[0,0]]])
        self.assertEqual(naive['pullback_fisher'],[[Q(3,16)]])
        self.assertEqual(naive['parameter_gradient'],[-Q(9,32)])
        # The naive score changes the context masses, not just the conditional policies.
        dp=naive['natural']['probability_tangent']
        self.assertNotEqual(sum(dp[:2]),0);self.assertEqual(sum(dp),0)

    def test_conditional_variance_not_total_joint_variance(self):
        x=audit_contexts(self.weights,self.contexts)
        self.assertEqual(x['expected_reward'],Q(1,8))
        q=[Q(3,8),Q(3,8),Q(1,8),Q(1,8)];r=[0,1,0,-2]
        total=sum(p*(v-x['expected_reward'])**2 for p,v in zip(q,r))
        self.assertEqual(total,Q(55,64))
        self.assertNotEqual(total,x['full_conditional_target_length_squared'])

    def test_single_context_matches_existing_audit(self):
        for key in ['natural','ordinary','diagonal','damped']:
            x=audit_contexts([1],self.contexts[:1]);a=audit(**self.contexts[0])
            self.assertEqual(x[key]['parameter_direction'],a[key]['parameter_direction'])
            self.assertEqual(x[key]['contexts'][0]['probability_tangent'],a[key]['probability_tangent'])
            self.assertEqual(x[key]['objective_derivative'],a[key]['objective_derivative'])

    def test_independent_weighted_score_and_centered_joint_reference(self):
        # Reconstruct from joint outcome scores directly, without production Gram helpers.
        contexts=[{'probabilities':['1/3','2/3'],'rewards':[-1,2],'coordinates':[[1,2,0],[0,-1,1]]},
                  {'probabilities':['1/2','1/3','1/6'],'rewards':[0,3,-2],'coordinates':[[1,0,1],[-1,2,0],[2,-1,1]]}]
        w=[Q(2,5),Q(3,5)];x=audit_contexts(w,contexts)
        samples=[];joint_p=[];joint_r=[];centered=[]
        for weight,c in zip(w,contexts):
            p=list(map(Q,c['probabilities']));a=c['coordinates'];means=[sum(p[i]*a[i][j] for i in range(len(p))) for j in range(3)]
            for i in range(len(p)):
                score=[Q(a[i][j])-means[j] for j in range(3)]
                samples.append((weight*p[i],Q(c['rewards'][i]),score));joint_p.append(weight*p[i]);joint_r.append(c['rewards'][i]);centered.append(score)
        g=[sum(q*r*s[j] for q,r,s in samples) for j in range(3)]
        metric=[[sum(q*s[i]*s[j] for q,r,s in samples) for j in range(3)] for i in range(3)]
        self.assertEqual(x['parameter_gradient'],g);self.assertEqual(x['pullback_fisher'],metric)
        joint=audit(joint_p,joint_r,centered)
        self.assertEqual(joint['pullback_fisher'],metric);self.assertEqual(joint['parameter_gradient'],g)
        self.assertEqual(x['natural']['parameter_direction'],joint['natural']['parameter_direction'])

    def test_weighted_objective_and_kl_finite_differences(self):
        contexts=[dict(self.contexts[0],coordinates=[[1,0],[0,2]]),dict(self.contexts[1],coordinates=[[0,-1],[2,0]])]
        x=audit_contexts(self.weights,contexts);h=0.0002
        def values(t):
            objective=kl=0.
            for w,c in zip([.75,.25],contexts):
                probs=[float(Q(p)) for p in c['probabilities']]
                logits=[math.log(p)+sum(float(a)*b for a,b in zip(row,t)) for p,row in zip(probs,c['coordinates'])]
                ex=[math.exp(z-max(logits)) for z in logits];new=[a/sum(ex) for a in ex]
                objective+=w*sum(q*r for q,r in zip(new,c['rewards']))
                kl+=w*sum(p*math.log(p/q) for p,q in zip(probs,new))
            return objective,kl
        for i in range(2):
            plus=[0.,0.];minus=[0.,0.];plus[i]=h;minus[i]=-h
            self.assertAlmostEqual((values(plus)[0]-values(minus)[0])/(2*h),float(x['parameter_gradient'][i]),places=7)
            for j in range(2):
                points=[]
                for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    t=[0.,0.];t[i]+=a*h;t[j]+=b*h;points.append(values(t)[1])
                fd=(points[0]-points[1]-points[2]+points[3])/(4*h*h)
                self.assertAlmostEqual(fd,float(x['pullback_fisher'][i][j]),places=6)

    def test_context_permutation_and_weight_splitting(self):
        x=audit_contexts(self.weights,self.contexts)
        reverse=audit_contexts(self.weights[::-1],self.contexts[::-1])
        split=audit_contexts(['1/4','1/2','1/4'],[self.contexts[0],self.contexts[0],self.contexts[1]])
        for y in [reverse,split]:
            for k in ['pullback_fisher','parameter_gradient','expected_reward','full_conditional_target_length_squared']:
                self.assertEqual(x[k],y[k])
            self.assertEqual(x['natural']['parameter_direction'],y['natural']['parameter_direction'])
            self.assertEqual(x['natural']['residual_length_squared'],y['natural']['residual_length_squared'])

    def test_each_context_reward_baseline_and_logit_gauge(self):
        x=audit_contexts(self.weights,self.contexts)
        changed=copy.deepcopy(self.contexts)
        for i,c in enumerate(changed):
            c['rewards']=[r+13*(i+1) for r in c['rewards']]
            c['coordinates']=[[v-7*(i+1) for v in row] for row in c['coordinates']]
        y=audit_contexts(self.weights,changed)
        for k in ['pullback_fisher','parameter_gradient','full_conditional_target_length_squared']:
            self.assertEqual(x[k],y[k])
        self.assertEqual(x['natural']['parameter_direction'],y['natural']['parameter_direction'])
        self.assertEqual(x['natural']['objective_derivative'],y['natural']['objective_derivative'])
        self.assertEqual(x['natural']['residual_length_squared'],y['natural']['residual_length_squared'])

    def test_seeded_exact_projection_optimality_and_nulls(self):
        rng=random.Random(61)
        for nc,d in [(1,1),(2,3),(3,2),(4,5),(6,12)]:
            w=[Q(i+1,nc*(nc+1)//2) for i in range(nc)]
            contexts=[]
            for i in range(nc):
                k=2+i%3
                contexts.append({'probabilities':[Q(j+1,k*(k+1)//2) for j in range(k)],
                    'rewards':[rng.randint(-3,3) for _ in range(k)],
                    'coordinates':[[rng.randint(-2,2) for _ in range(d)] for _ in range(k)]})
            x=audit_contexts(w,contexts);n=x['natural']
            self.assertTrue(x['gram_agrees']);self.assertEqual(x['solve_residual'],[0]*d)
            self.assertEqual(n['projection_inner_product'],0);self.assertEqual(n['pythagorean_residual'],0)
            self.assertEqual(n['objective_derivative'],n['fisher_length_squared'])
            self.assertEqual(n['probability_length_squared'],n['fisher_length_squared'])
            self.assertGreaterEqual(n['objective_derivative'],0)
            for null in x['null_probability_tangents']:
                for c,v in zip(contexts,null):self.assertEqual(v,[0]*len(c['rewards']))
            # Brute sampled competitors cannot beat the exact weighted metric projection.
            for _ in range(8):
                v=[Q(rng.randint(-9,9),7) for _ in range(d)]
                length=Q(0)
                for q,local in zip(w,x['local_audits']):
                    dp=[sum(row[j]*v[j] for j in range(d)) for row in local['probability_jacobian']]
                    length+=q*sum((t-a)**2/p for t,a,p in zip(local['full_simplex_natural_tangent'],dp,local['probabilities']))
                self.assertGreaterEqual(length,n['residual_length_squared'])
            # Weighted positive pair certificate yields the aggregate quadratic exactly.
            v=[Q(rng.randint(-3,3),5) for _ in range(d)]
            gram=sum(v[i]*x['pullback_fisher'][i][j]*v[j] for i in range(d) for j in range(d))
            squares=sum(t['weight']*sum(a*b for a,b in zip(t['difference'],v))**2 for t in x['pair_certificate'])
            self.assertEqual(gram,squares);self.assertGreaterEqual(gram,0)

    def test_context_validation(self):
        for w,c in [([],[]),([1],None),([1]*7,self.contexts*3+[self.contexts[0]]),([],self.contexts),
                    ([0,1],self.contexts),([-1,2],self.contexts),(['1/2','1/4'],self.contexts),
                    ([True,False],self.contexts),([.75,.25],self.contexts),([1],[dict(self.contexts[0],extra=1)]),
                    ([1],[{}]),([1],[None]),(self.weights,[self.contexts[0],dict(self.contexts[1],coordinates=[[0,0],[1,0]])])]:
            with self.subTest(args=(w,c)),self.assertRaises(ValueError):audit_contexts(w,c)

    def test_no_mutation_global_rng_and_repeated_output(self):
        before=copy.deepcopy(self.contexts);state=random.getstate()
        self.assertEqual(run_context_demo(),run_context_demo());audit_contexts(self.weights,self.contexts)
        self.assertEqual(before,self.contexts);self.assertEqual(state,random.getstate())

    def test_context_cli_and_committed_reports(self):
        def call(*args):return subprocess.run([sys.executable,'-m','fisherlab',*args],capture_output=True,text=True)
        out=call('context-demo');self.assertEqual(out.returncode,0,out.stderr)
        self.assertEqual(json.loads(out.stdout),encode(run_context_demo()))
        self.assertEqual(out.stdout,Path('examples/context-demo.json').read_text())
        out=call('contexts','examples/contexts-input.json');self.assertEqual(out.returncode,0,out.stderr)
        self.assertEqual(json.loads(out.stdout),encode(audit_contexts(self.weights,self.contexts)))
        base=json.loads(Path('examples/contexts-input.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for text in ['{"version":1,"version":1}',json.dumps(dict(base,version=True)),json.dumps(dict(base,extra=1)),json.dumps(dict(base,weights=[1,0])),json.dumps(dict(base,contexts=[dict(self.contexts[0],extra=1)])),json.dumps([])]:
                p.write_text(text);out=call('contexts',str(p));self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'')


if __name__=='__main__':unittest.main()
