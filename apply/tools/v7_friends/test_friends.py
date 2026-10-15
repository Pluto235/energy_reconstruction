"""Small synthetic checks; no physics files or model required."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from friends import match_rows, compare, native_sum, publish, MATCH_FIELDS

class Checks(unittest.TestCase):
    def table(self, values):
        return {k:np.asarray(values,dtype='f8') for k in MATCH_FIELDS}
    def test_order_and_precision(self):
        e=self.table([.123456789,2,3]); s={k:v[[2,0,1]].astype('f4') for k,v in e.items()}
        rows,stats=match_rows(e,s)
        np.testing.assert_array_equal(rows,[1,2,0]); self.assertEqual(stats['missing'],0)
        self.assertEqual(compare(e['theta'],s['theta'][rows])['outside_rounding'],0)
    def test_duplicate_missing_nonfinite(self):
        for a,b,field in [([1,1],[1,2],'event_duplicate_rows'),([1,2],[1,1],'summary_duplicate_rows'),([1,3],[1,2],'missing'),([1,float('nan')],[1,2],'nonfinite_event_rows')]:
            rows,stats=match_rows(self.table(a),self.table(b))
            self.assertIsNone(rows); self.assertGreater(stats[field],0)
    def test_rounding_not_tolerance(self):
        x=np.array([1.123456789],dtype='f8'); y=x.astype('f4')
        self.assertEqual(compare(x,y)['rounding_only'],1)
        y=np.nextafter(y,np.float32(np.inf)); self.assertEqual(compare(x,y)['outside_rounding'],1)
    def test_integer_sum_and_overflow(self):
        s={f'nq03t30s{i}':np.array([2**31-1],dtype='i4') for i in range(3)}
        self.assertEqual(native_sum(s)[0],3*(2**31-1))
        s['nq03t30s0']=np.array([-1],dtype='i4')
        with self.assertRaises(ValueError): native_sum(s)
        s={f'nq03t30s{i}':np.array([2**63-1],dtype='i8') for i in range(3)}
        with self.assertRaises(ValueError): native_sum(s)
    def test_atomic_no_publish_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'out'
            def fail(p):
                (p/'partial').write_text('bad')
                raise ValueError('readback failed')
            with self.assertRaises(ValueError): publish(target,fail)
            self.assertFalse(target.exists())
            publish(target,lambda p:(p/'receipt.json').write_text('{}'))
            with self.assertRaises(FileExistsError): publish(target,lambda p:None)

if __name__=='__main__': unittest.main()
