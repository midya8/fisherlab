import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from fisherlab import encode, run_context_demo
spec=importlib.util.spec_from_file_location('render_contexts','examples/render_contexts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ContextFigureTests(unittest.TestCase):
    def test_all_chart_cells_positions_and_controls(self):
        r=encode(run_context_demo());root=ET.fromstring(module.render(r));cells={e.attrib['id']:e for e in root.iter() if 'id' in e.attrib}
        keys=['weighted_shared','weighted_independent','uniform_shared_different_objective']
        labels=['weighted, shared','weighted, independent','uniform, shared (new objective)']
        xs=[44,321,445,580,715,848,985]
        for i,key in enumerate(keys):
            a=r[key];n=a['natural'];vals=[labels[i],f"{a['system']['rank']} / {len(n['parameter_direction'])}",','.join(n['parameter_direction']),n['contexts'][0]['reward_derivative'],n['contexts'][1]['reward_derivative'],n['objective_derivative'],n['residual_length_squared']]
            for j,v in enumerate(vals):
                c=cells[f'chart-{i}-{j}'];self.assertEqual(c.text,v);self.assertEqual(c.attrib['x'],str(xs[j]));self.assertEqual(c.attrib['y'],str(302+43*i))
        self.assertEqual(cells['pyth-equation'].text,'7/16 = 1/64 + 27/64')
        self.assertEqual(cells['projection'].text,'0');self.assertEqual(cells['pyth-residual'].text,'0');self.assertEqual(cells['wrong-objective'].text,'-1/32')
        self.assertEqual([cells[f'gauge-{i}'].text for i in range(4)],['0','0','3/16','-9/32'])

    def test_complete_witness_and_invisible_tamper(self):
        r=run_context_demo();svg=module.render(r);root=ET.fromstring(svg)
        m=next(e for e in root.iter() if e.attrib.get('id')=='full-witness')
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(m.text))),encode(r))
        altered=copy.deepcopy(r);altered['weighted_shared']['local_audits'][0]['score_rows'][0][0]+=1
        self.assertNotEqual(module.render(altered),svg)

    def test_committed_svg_matches_fresh_report(self):
        self.assertEqual(Path('assets/context-audit.svg').read_text(),module.render(run_context_demo()))
