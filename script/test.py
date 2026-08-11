torch_data = """
"""


dtorch_data = """
"""


def parser(datas):
    result = []
    for data in datas.splitlines():
        if not data:
            continue
        time_data_str = data.split("\t")[2]
        time_data_str = time_data_str.split(" ")
        time_data = time_data_str[0]
        time_data_unit = time_data_str[1]
        if time_data_unit == "ms":
            time_float = float(time_data)
        elif time_data_unit == "μs":
            continue
            time_float = float(time_data) / 1000
        else:
            raise ValueError(f"Invalid time data unit: {time_data_unit}")
        result.append(time_float)
    return result


torch_data = parser(torch_data)
dtorch_data = parser(dtorch_data)

print(f"torch_data, length: {len(torch_data)}, ave: {sum(torch_data) / len(torch_data)}")
print(f"dtorch_data, length: {len(dtorch_data)}, ave: {sum(dtorch_data) / len(dtorch_data)}")
