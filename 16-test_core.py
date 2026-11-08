import copy
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from fisherlab import audit, encode, run_demo
from fisherlab.core import rational, solve
from fisherlab.demo import nonlinear_demo

P = [Q(1,2), Q(1,3), Q(1,6)]
R = [Q(0), Q(1), Q(3)]
A = [[1,2], [0,1], [0,0]]


def weighted_prob(t, p=P, a=A):
    # Independent finite softmax through log(p)+A*t, without core matrix helpers.
    e = [math.exp(math.log(float(p[i]))+sum(float(a[i][j])*t[j] for j in range(len(t)))) for i in range(len(p))]
    return [v/sum(e) for v in e]


def mean(t, p=P, r=R, a=A):
    return sum(float(r[i])*q for i,q in enumerate(weighted_prob(t,p,a)))


class GeometryTests(unittest.TestCase):
    def test_closed_form_logit_fisher_and_target(self):
        x=audit(P,R,[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(x['logit_fisher'],[[Q(1,4),-Q(1,6),-Q(1,12)],[-Q(1,6),Q(2,9),-Q(1,18)],[-Q(1,12),-Q(1,18),Q(5,36)]])
        self.assertEqual(x['natural']['probability_tangent'],[-Q(5,12),Q(1,18),Q(13,36)])
        self.assertEqual(x['reward_variance'],Q(41,36))
        self.assertEqual(x['expected_reward'],Q(5,6))

    def test_score_outer_product_independent(self):
        x=audit(P,R,A)
        s=[[Q(A[i][v])-sum(P[h]*A[h][v] for h in range(3)) for v in range(2)] for i in range(3)]
        f=[[sum(P[i]*s[i][u]*s[i][v] for i in range(3)) for v in range(2)] for u in range(2)]
        self.assertEqual(x['score_rows'],s)
        self.assertEqual(x['pullback_fisher'],f)
        self.assertEqual(x['parameter_gradient'],[sum(P[i]*R[i]*s[i][v] for i in range(3)) for v in range(2)])
        self.assertEqual([sum(P[i]*s[i][v] for i in range(3)) for v in range(2)],[0,0])

    def test_independent_probability_and_reward_derivatives(self):
        x=audit(P,R,A); h=1e-5
        for j in range(2):
            plus=[0.,0.];minus=[0.,0.];plus[j]=h;minus[j]=-h
            self.assertAlmostEqual((mean(plus)-mean(minus))/(2*h),float(x['parameter_gradient'][j]),places=8)
            for i in range(3):
                observed=(weighted_prob(plus)[i]-weighted_prob(minus)[i])/(2*h)
                self.assertAlmostEqual(observed,float(x['probability_jacobian'][i][j]),places=8)

    def test_independent_kl_hessian_all_entries(self):
        x=audit(P,R,A); h=0.0003
        def kl(t):
            return sum(float(p)*math.log(float(p)/q) for p,q in zip(P,weighted_prob(t)))
        for u,v in itertools.product(range(2),repeat=2):
            pp=[0.,0.];pm=[0.,0.];mp=[0.,0.];mm=[0.,0.]
            pp[u]+=h;pp[v]+=h;pm[u]+=h;pm[v]-=h;mp[u]-=h;mp[v]+=h;mm[u]-=h;mm[v]-=h
            fd=(kl(pp)-kl(pm)-kl(mp)+kl(mm))/(4*h*h)
            self.assertAlmostEqual(fd,float(x['pullback_fisher'][u][v]),places=6)

    def test_exact_full_charts_all_induced_tangents(self):
        demo=run_demo()
        for row in demo['charts'][:4]:
            x=row['audit'];natural=x['natural']
            self.assertTrue(x['full_simplex_reachable'])
            self.assertEqual(natural['probability_tangent'],x['full_simplex_natural_tangent'])
            self.assertEqual(natural['fisher_length_squared'],x['reward_variance'])
            self.assertEqual(natural['objective_derivative'],x['reward_variance'])
            self.assertEqual(natural['probability_length_squared'],x['reward_variance'])
            self.assertTrue(x['gram_agrees'])

    def test_rref_not_minimum_norm_and_null_witness(self):
        x=audit(P,R,[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(x['natural']['parameter_direction'],[-3,-2,0])
        self.assertEqual(x['system']['null_basis'],[[1,1,1]])
        self.assertEqual(x['null_probability_tangents'],[[0,0,0]])
        self.assertEqual(x['solve_residual'],[0,0,0])
        self.assertNotEqual(sum(x['natural']['parameter_direction']),0)

    def test_redundant_null_vectors_all_invisible(self):
        x=run_demo()['charts'][3]['audit']
        self.assertEqual(x['system']['rank'],2)
        self.assertEqual(len(x['system']['null_basis']),2)
        for v, dp in zip(x['system']['null_basis'],x['null_probability_tangents']):
            self.assertEqual(dp,[0,0,0])
            self.assertEqual(sum(g*q for g,q in zip(x['parameter_gradient'],v)),0)

    def test_restricted_projection_exact_and_orthogonality(self):
        x=audit(P,R,[[1],[0],[0]])
        self.assertFalse(x['full_simplex_reachable'])
        self.assertEqual(x['natural']['parameter_direction'],[-Q(5,3)])
        self.assertEqual(x['natural']['probability_tangent'],[-Q(5,12),Q(5,18),Q(5,36)])
        self.assertEqual(x['natural']['full_simplex_gap_l1'],Q(4,9))
        residual=[a-b for a,b in zip(x['full_simplex_natural_tangent'],x['natural']['probability_tangent'])]
        self.assertEqual(sum(dp*e/p for dp,e,p in zip(x['natural']['probability_tangent'],residual,P)),0)
        self.assertLess(x['natural']['objective_derivative'],x['reward_variance'])

    def test_zero_chart_is_frozen_not_full_rank(self):
        x=audit(P,R,[[0,0],[0,0],[0,0]])
        self.assertEqual(x['system']['rank'],0)
        self.assertFalse(x['full_simplex_reachable'])
        for key in ['natural','ordinary','diagonal','damped']:
            self.assertEqual(x[key]['probability_tangent'],[0,0,0])
        self.assertEqual(x['system']['null_basis'],[[1,0],[0,1]])

    def test_constant_gauge_chart_frozen(self):
        x=audit(P,R,[[2,-1],[2,-1],[2,-1]])
        self.assertEqual(x['pullback_fisher'],[[0,0],[0,0]])
        self.assertEqual(x['natural']['probability_tangent'],[0,0,0])

    def test_diagonal_and_damping_break_chart_agreement(self):
        b=audit(P,R,[[1,0],[0,1],[0,0]]);s=audit(P,R,A)
        self.assertEqual(b['diagonal']['full_simplex_gap_l1'],Q(5,9))
        self.assertEqual(s['diagonal']['full_simplex_gap_l1'],Q(14,15))
        self.assertEqual(b['damped']['full_simplex_gap_l1'],Q(7,17))
        self.assertEqual(s['damped']['full_simplex_gap_l1'],Q(43,213))
        for key in ['ordinary','diagonal','damped']:
            self.assertNotEqual(b[key]['probability_tangent'],s[key]['probability_tangent'])

    def test_additive_gauge_columns_and_reward_shift(self):
        x=audit(P,R,A)
        y=audit(P,[v+17 for v in R],[[v+Q(7,3)*(h+1) for h,v in enumerate(row)] for row in A])
        for key in ['pullback_fisher','parameter_gradient','probability_jacobian','reward_variance','natural','ordinary','diagonal','damped']:
            # logit directions can differ by an invisible common constant.
            if isinstance(x[key],dict):
                self.assertEqual(x[key]['probability_tangent'],y[key]['probability_tangent'])
            else:self.assertEqual(x[key],y[key])

    def test_action_and_coordinate_permutations(self):
        x=audit(P,R,A);perm=[2,0,1]
        y=audit([P[i] for i in perm],[R[i] for i in perm],[A[i][::-1] for i in perm])
        for key in ['natural','ordinary','diagonal','damped']:
            self.assertEqual(y[key]['probability_tangent'],[x[key]['probability_tangent'][i] for i in perm])

    def test_constant_rewards_zero_every_direction(self):
        x=audit(P,[5,5,5],A)
        self.assertEqual(x['reward_variance'],0)
        for key in ['natural','ordinary','diagonal','damped']:
            self.assertEqual(x[key]['parameter_direction'],[0,0])
            self.assertEqual(x[key]['objective_derivative'],0)

    def test_seeded_random_charts_exact_certificates(self):
        rng=random.Random(73)
        for k in [2,3,5,8]:
            p=[Q(i+1,k*(k+1)//2) for i in range(k)]
            r=[Q(rng.randint(-9,9),3) for _ in range(k)]
            for d in [1,k-1,k,12]:
                a=[[rng.randint(-2,2) for _ in range(d)] for _ in range(k)]
                x=audit(p,r,a)
                self.assertTrue(x['gram_agrees'])
                self.assertEqual(x['solve_residual'],[0]*d)
                self.assertTrue(all(v==[0]*k for v in x['null_probability_tangents']))
                n=x['natural']
                self.assertEqual(sum(n['probability_tangent']),0)
                self.assertEqual(n['fisher_length_squared'],n['probability_length_squared'])
                self.assertEqual(n['objective_derivative'],n['fisher_length_squared'])
                self.assertGreaterEqual(n['objective_derivative'],0)
                self.assertLessEqual(n['objective_derivative'],x['reward_variance'])
                if x['full_simplex_reachable']:self.assertEqual(n['full_simplex_gap_l1'],0)
                # Independent explicit sum-of-squares certificate for each test vector.
                v=[Q(rng.randint(-5,5),7) for _ in range(d)]
                quadratic=sum(v[i]*x['pullback_fisher'][i][j]*v[j] for i in range(d) for j in range(d))
                sos=sum(pair['weight']*sum(t*z for t,z in zip(v,pair['difference']))**2 for pair in x['pair_certificate'])
                self.assertEqual(quadratic,sos)
                self.assertGreaterEqual(sos,0)

    def test_full_affine_chart_finite_updates_agree(self):
        x=run_demo()
        p0=None
        for row in x['charts'][:4]:
            direction=row['audit']['natural']['logit_direction']
            values=[math.log(float(p))+0.25*float(v) for p,v in zip(P,direction)]
            ex=[math.exp(v-max(values)) for v in values];new=[v/sum(ex) for v in ex]
            if p0 is None:p0=new
            for a,b in zip(new,p0):self.assertAlmostEqual(a,b,places=14)
        # Specific to affine-logit charts; the nonlinear fixture does not share this property.

    def test_nonlinear_same_tangent_different_finite_step(self):
        x=nonlinear_demo()
        self.assertEqual(x['mapped_logit_tangent'],x['natural_logit_direction'])
        self.assertEqual(x['direct_euler_logit'],Q(3,2))
        self.assertEqual(x['curved_euler_logit'],Q(343,216))
        self.assertEqual(x['logit_difference'],Q(19,216))
        self.assertGreater(x['curved_probability_action1'],x['direct_probability_action1'])
        # Error is genuinely second order in eta for the cubic map at theta1.
        for eta in [Q(1,100),Q(1,10),Q(1,2)]:
            self.assertEqual((1+eta/3)**3-(1+eta),eta*eta/3+eta**3/27)

    def test_local_kl_not_finite_bound(self):
        rows=run_demo()['finite_steps']
        ratios=[row['actual_kl_old_to_new']/float(row['quadratic_kl_prediction']) for row in rows]
        self.assertLess(abs(ratios[0]-1),0.01)
        self.assertGreater(abs(ratios[-1]-1),0.2)
        self.assertNotEqual(rows[-1]['expected_reward'],float(rows[-1]['linear_reward_prediction']))
        for row in rows:self.assertAlmostEqual(sum(row['updated_probabilities']),1,places=15)

    def test_no_mutation_or_global_rng(self):
        state=random.getstate();args=[P.copy(),R.copy(),copy.deepcopy(A)];before=copy.deepcopy(args)
        audit(*args);run_demo()
        self.assertEqual(args,before);self.assertEqual(state,random.getstate())

    def test_strict_rationals_and_limits(self):
        for value in [True,False,0.5,float('nan'),float('inf'),None,{},[], '1/0','NaN','', '1'*101,2**128]:
            with self.subTest(value=value),self.assertRaises(ValueError):rational(value)
        self.assertEqual(rational('0.125'),Q(1,8))
        self.assertEqual(rational('-2/3'),-Q(2,3))
        x=encode(audit(P,R,A));self.assertEqual(x['expected_reward'],'5/6')
        self.assertEqual(x['system']['rank'],2)

    def test_invalid_probability_reward_chart_damping(self):
        bad=[([],R,A,'1/10'),([1],R,A,'1/10'),([0,Q(1,2),Q(1,2)],R,A,'1/10'),
             ([-1,1,1],R,A,'1/10'),([1,1,1],R,A,'1/10'),(P,[1],A,'1/10'),
             (P,R,[[1],[1]],'1/10'),(P,R,[[],[],[]],'1/10'),(P,R,[[1],[1,2],[0]],'1/10'),
             (P,R,[[0]*13 for _ in P],'1/10'),(P,R,A,0),(P,R,A,-1),
             (P,R,[1,2,3],'1/10'),(P,R,None,'1/10'),(P,R,A,True)]
        for p,r,a,l in bad:
            with self.subTest(args=(p,r,a,l)),self.assertRaises(ValueError):audit(p,r,a,l)

    def test_solver_consistent_inconsistent_and_shape(self):
        x=solve([[1,2],[2,4]],[3,6])
        self.assertEqual(x['solution'],[3,0]);self.assertEqual(x['null_basis'],[[-2,1]])
        self.assertEqual(x['rref_augmented'],[[1,2,3],[0,0,0]])
        for a,b in [([[1,2],[2,4]],[3,7]),([],[]),([[1,2]],[1])]:
            with self.assertRaises(ValueError):solve(a,b)


class CLITests(unittest.TestCase):
    def call(self,*args):
        return subprocess.run([sys.executable,'-m','fisherlab',*args],capture_output=True,text=True)

    def test_demo_and_input_cli(self):
        demo=self.call('demo');self.assertEqual(demo.returncode,0,demo.stderr)
        self.assertEqual(json.loads(demo.stdout),encode(run_demo()))
        result=self.call('audit','examples/input.json');self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),encode(audit(P,R,A)))

    def test_bad_files_and_usage(self):
        for args in [(),('bad',),('demo','extra'),('audit','not-found-file')]:
            result=self.call(*args);self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,'')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json'
            base={'version':1,'probabilities':['1/2','1/3','1/6'],'rewards':[0,1,3],'coordinates':A,'damping':'1/10'}
            bad=[{},[],dict(base,version=True),dict(base,extra=1),dict(base,probabilities=[.5,.25,.25]),dict(base,damping=0)]
            texts=['{bad','{"version":1,"version":1}', ' '*100001]+[json.dumps(x) for x in bad]
            for text in texts:
                path.write_text(text);out=self.call('audit',str(path))
                self.assertEqual(out.returncode,2,out.stderr);self.assertEqual(out.stdout,'')

    def test_committed_demo_reproducible(self):
        self.assertEqual(json.loads(Path('examples/demo.json').read_text()),encode(run_demo()))


if __name__=='__main__':unittest.main()
