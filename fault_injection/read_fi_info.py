import struct
import numpy as np

FILENAME = "fi_info.bin"

def read_records(filename):
    records = []
    with open(filename, "rb") as f:
        while True:
            header = f.read(4)  # int32 step
            if not header or len(header) < 4:
                break
            step = struct.unpack('<i', header)[0]  # little-endian
            
            code_bytes = f.read(2)
            if len(code_bytes) < 2:
                raise EOFError("unexpected EOF when reading code")
            code = code_bytes.decode('ascii')
            
            count_bytes_1 = f.read(4)
            if len(count_bytes_1) < 4:
                raise EOFError("unexpected EOF when reading count")
            count_1 = struct.unpack('<i', count_bytes_1)[0]

            # read count floats
            float_bytes_1 = f.read(4 * count_1)
            if len(float_bytes_1) < 4 * count_1:
                raise EOFError("unexpected EOF when reading floats")
            # faster: use numpy.frombuffer
            floats_1 = np.frombuffer(float_bytes_1, dtype=np.float32).copy()

            count_bytes_2 = f.read(4)
            if len(count_bytes_2) < 4:
                raise EOFError("unexpected EOF when reading count")
            count_2 = struct.unpack('<i', count_bytes_2)[0]

            # read count floats
            float_bytes_2 = f.read(4 * count_2)
            if len(float_bytes_2) < 4 * count_2:
                raise EOFError("unexpected EOF when reading floats")
            # faster: use numpy.frombuffer
            floats_2 = np.frombuffer(float_bytes_2, dtype=np.float32).copy()

            records.append({
                'step': step,
                'code': code,
                'count_1': count_1,
                'floats_1': floats_1,
                'count_2': count_2,
                'floats_2': floats_2
            })
    return records

if __name__ == "__main__":
    recs = read_records(FILENAME)
    for i, r in enumerate(recs):
        print(f"Record {i}: step={r['step']}, code={r['code']}, count1={r['count_1']}, floats1={r['floats_1']}, count2={r['count_2']}, floats2={r['floats_2']}")
