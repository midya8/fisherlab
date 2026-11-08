import base64
import copy
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from fisherlab import encode, run_demo

spec=importlib.util.spec_from_file_location('render_audit','examples/render_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FigureTests(unittest.TestCase):
    def test_all_table_values_and_positions(self):
        r=encode(run_demo());root=ET.fromstring(module.render(r))
        cells={x.attrib['id']:x for x in root.iter() if 'id' in x.attrib}
        columns=[42,205,348,455,565,680,815,950]
        for i,row in enumerate(r['charts']):
            a=row['audit'];values=[row['name'],f"{a['system']['rank']} / {len(a['coordinates'][0])}",*a['natural']['probability_tangent'],a['natural']['full_simplex_gap_l1'],a['diagonal']['full_simplex_gap_l1'],a['damped']['full_simplex_gap_l1']]
            for j,v in enumerate(values):
                cell=cells[f'chart-{i}-{j}'];self.assertEqual(cell.text,str(v))
                self.assertEqual(cell.attrib['x'],str(columns[j]));self.assertEqual(cell.attrib['y'],str(210+37*i))
        for i,row in enumerate(r['finite_steps']):
            values=[row['step'],f"{row['actual_kl_old_to_new']:.6f}",module.number(row['quadratic_kl_prediction']),f"{row['expected_reward']:.6f}",module.number(row['linear_reward_prediction'])]
            for j,v in enumerate(values):
                cell=cells[f'step-{i}-{j}'];self.assertEqual(cell.text,v)
                self.assertEqual(cell.attrib['y'],str(762+29*i));self.assertEqual(cell.attrib['x'],str([43,190,392,620,850][j]))
        self.assertEqual(cells['nonlinear-2'].text,'19/216')
        self.assertEqual(cells['nonlinear-1'].text,'3/2 / 343/216')

    def test_full_witness_metadata_and_hidden_change(self):
        r=run_demo();svg=module.render(r);root=ET.fromstring(svg)
        m=next(x for x in root.iter() if x.attrib.get('id')=='full-witness')
        decoded=json.loads(zlib.decompress(base64.b64decode(m.text)))
        self.assertEqual(decoded,encode(r))
        altered=copy.deepcopy(r);altered['charts'][0]['audit']['system']['null_basis'].append([Q(99),Q(99)])
        self.assertNotEqual(module.render(altered),svg)

    def test_committed_svg_byte_exact(self):
        self.assertEqual(Path('assets/fisher-audit.svg').read_text(),module.render(run_demo()))


if __name__=='__main__':unittest.main()
