"""
Test data
"""
import json
def test_data():
    """
    Test data function.
    """
 
    with open("/Users/cavelab/Documents/Github/eager_project_oop/eager_project_oop/.cache/d1_20181029_0800_0830_next_timestamp_occupancy_682a48de_60f06e613772749d2f075c50497a9c48.json", "r") as f:
        data = json.load(f)
    print(data)
    assert False