import unittest
import pandas as pd
import pyarrow as pa


class TestUISchemas(unittest.TestCase):
    def test_year_column_pyarrow_serialization(self):
        # Emulate rows with integer years and missing ("—") years
        data = [
            {"Title": "Paper A", "Year": "2023", "Citations": 10},
            {"Title": "Paper B", "Year": "—", "Citations": 5},
            {"Title": "Paper C", "Year": "2024", "Citations": 100},
        ]
        df = pd.DataFrame(data)
        # Converting to PyArrow table must succeed without ArrowInvalid
        arrow_table = pa.Table.from_pandas(df)
        self.assertEqual(len(arrow_table), 3)

    def test_chunks_column_pyarrow_serialization(self):
        # Emulate rows with string chunks "5", "—", etc.
        data = [
            {"ID": "1706.03762", "Chunks": "15"},
            {"ID": "2401.00001", "Chunks": "—"},
        ]
        df = pd.DataFrame(data)
        arrow_table = pa.Table.from_pandas(df)
        self.assertEqual(len(arrow_table), 2)


if __name__ == "__main__":
    unittest.main()
