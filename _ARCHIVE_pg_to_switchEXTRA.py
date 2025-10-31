import pandas as pd
import ast
gen_info = pd.read_csv('/Users/melek/Switch-USA-PG/switch/26-zone/in_noCluster2/2050/base_short/gen_info2.csv')
powerplants = pd.read_csv('/Users/melek/Downloads/power-plants0.csv')
# gen_info['plant_id'] = gen_info['unit_id_pg'].apply(extract_plant_ids)
# 1) build a boolean mask: True whenever unit_id_pg is a list or a string that parses to a list
is_list_or_list_str = gen_info['unit_id_pg'].apply(
    lambda x: isinstance(x, list)
              or (isinstance(x, str)
                  and x.strip().startswith('[')
                  and x.strip().endswith(']'))
)

# 2) apply extract_plant_ids ONLY on those rows, and write back into plant_id
gen_info.loc[is_list_or_list_str, 'plant_id'] = (
    gen_info.loc[is_list_or_list_str, 'unit_id_pg']
            .apply(extract_plant_ids)
)

def extract_plant_ids(x):
    """
    - If x is a Python list (e.g. ['59817_BATT1', '60654_GEN02', ...]), 
      convert each element to str, split on '_', take the left part, 
      then cast to float→int (to handle things like "59817.0" if it ever appears).
    - Else if x is a string that literal‐evaluates to a list, do the same.
    - Otherwise (float, int, <NA>), return x unchanged.
    """
    # Case A: x is already a list
    if isinstance(x, list):
        out = []
        for item in x:
            # Force item to string, split on "_", take the first chunk,
            # then cast to float→int so that "59817.0" → 59817 without error.
            first_chunk = str(item).split('_', 1)[0]
            try:
                out.append(int(float(first_chunk)))
            except ValueError:
                # if somehow it isn’t numeric, skip or keep as NA
                continue
        return out

    # Case B: x is a string that looks like a Python list (e.g. "['59817_BATT1', …]")
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    first_chunk = str(item).split('_', 1)[0]
                    try:
                        out.append(int(float(first_chunk)))
                    except ValueError:
                        continue
                return out
        except (ValueError, SyntaxError):
            # x wasn’t a literal‐evaluatable list, so fall through
            return x
    

    # Case C: anything else (float, int, <NA>, etc.), leave it unchanged
    return x


