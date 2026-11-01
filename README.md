# metadata-tools

Memory-bounded reading and data dictionaries for CSV, Stata (`.dta`) and SPSS (`.sav`, `.zsav`) files
in chunks. Built to process large survey files on a server with limited RAM.

Reading only for now. Writing will be added later.

> Code in this repository was generated with the help of AI.

```python
from metadata_tools import open_reader, MiB

with open_reader("survey.sav", chunk_memory_bytes=256 * MiB) as reader:
    print(reader.meta.number_rows, reader.meta.column_labels)
    for chunk in reader.iter_chunks():   # pandas DataFrames
        ...
```

## Data dictionary

`get_dictionary(path)` builds a dictionary from the file's metadata only:
names, labels, types, formats, categories and declared missing values. It does
not compute frequencies or summary statistics. It never reads data rows, so it
takes about 0.01 s even for a 1 GB file.

```python
from metadata_tools import get_dictionary

d = get_dictionary("survey.sav")
d.to_dict()   # plain dict, ready for json.dumps
```

```json
{
  "path": "survey.sav", "format": "spss", "encoding": "UTF-8",
  "rows": 200, "columns": 5, "file_label": "...", "notes": [],
  "created": "2026-10-17T12:35:40", "modified": "2026-10-17T12:35:40",
  "variables": [
    {
      "name": "q2", "label": "Nominal with user missings 8/9",
      "type": "numeric", "storage_type": "double", "format": "F8.2",
      "decimals": 2, "measure": "nominal",
      "categories": [
        {"value": 1, "label": "Yes", "is_missing": false},
        {"value": 8, "label": "Don't know", "is_missing": true}
      ],
      "missing_values": [8, 9], "missing_ranges": []
    }
  ]
}
```

Field notes:

- **`type`** is `numeric`, `string`, `date`, `datetime` or `time`.
  Dates are recognised from the display format (`%td`, `DATE11`, ...).
- **Category values** are read the way they were entered: `1`, not `1.0`.
  Stata extended missing values appear as `.a`, `.b`, ... with
  `is_missing: true`.
- **`missing_values` and `missing_ranges`** are declared in SPSS files. Stata
  has no declared missing values.
- **`measure`** is set for SPSS only.

- **CSV files** store nothing but column names, so a CSV dictionary has names
  only. `type`, `storage_type`, `label` and `rows` are `null`. Guessing types
  from a sample can be wrong, and getting them right means reading every
  value, so the dictionary doesn't attempt it. Only the header line is
  decoded, plus the first 64 KiB to detect the encoding and delimiter. When
  you need column types, use `open_reader(path).meta`, which scans the
  whole file.

## Reading data

What `open_reader` gives you:

- **One interface for every format.** `open_reader(path)` returns a `Reader`
  with `.meta` (a `FileMeta`), `.iter_chunks()` and `.read()`.
- **Chunks sized by memory, not rows.** Each chunk is about
  `chunk_memory_bytes` in memory (default 256 MiB), measured from a sample
  of the file. Set `chunk_rows` to use a fixed row count instead.
- **Stable dtypes.** Every chunk has the dtypes listed in `meta.dtypes`, even
  when a chunk has no values in a date column or no missing values in an
  integer column. The chunk index continues across chunks.
- **Compact dtypes by default** (`dtype_backend="pyarrow"`). Strings are Arrow
  strings, and Stata `byte`/`int`/`float` columns keep their 1/2/4-byte width.
  Use `dtype_backend="numpy"` to get float64 for numbers and object for strings.
- **`read()` refuses to load files that are too large.** It raises
  `MemoryBudgetError` when its estimate is above `max_full_read_bytes`
  (default 2 GiB).
- **Completeness check.** `IncompleteReadError` is raised if fewer rows are
  read than the file header declares.
- **Decoding fallbacks**, carried over from metadata-editor-fastapi:
  - Several encodings are tried in turn, chunk by chunk, so a file that changes
    encoding partway through still reads.
  - Stata date conversion errors fall back to raw numbers.
  - Stata extended-missing `chr()` errors fall back to NaN.
  - Stata files pyreadstat cannot decode at all are read with pandas `StataReader`.
  - Every fallback taken is listed in `reader.warnings`.

## CSV

CSV files are read with `pyarrow.csv`:

- The encoding is detected by streaming the file through a decoder, and the
  delimiter is sniffed from the first 64 KiB.
- Quoted values may contain newlines.
- Integers with leading zeros (`"0042"`) and integers too large for int64 stay
  strings, so no digits are lost.

`csv_type_inference` controls column types:

| Mode | Behaviour |
|---|---|
| `scan` (default) | A streaming pass over the whole file decides each column's type before any data is yielded. It also gives an exact `meta.number_rows`. It costs about 4 s per 1.4 GB file. |
| `sample` | Types are inferred from the first block only. A later value that does not fit raises `SchemaDriftError`. |
| `string` | Every column is a string. |

`dtypes={"col": "int32"}` overrides the type of individual columns.

## Memory

Peak process memory is about **2–3 × `chunk_memory_bytes` + ~200 MB** for
imports. Measured on macOS with pandas 3.1, pyarrow 26 and pyreadstat 1.3.7:

| File | Full read (app today, pandas 2.2) | `chunk_memory_bytes` | Peak RSS | Time |
|---|---|---|---|---|
| CSV, 1.43 GB, 16M rows × 8 | 4,159 MB, 6.3 s | 64 MB | 393 MB | 7.1 s |
| | | 256 MB | 893 MB | 7.1 s |
| | | 1 GB | 1,610 MB | 7.2 s |
| SAV, 1.06 GB, 3M rows × 44 | 2,755 MB, 11.1 s | 64 MB | 457 MB | 7.4 s |
| | | 256 MB | 1,116 MB | 7.7 s |
| | | 1 GB | 2,787 MB | 8.1 s |

To pick a budget on the server, run:

```
uv run python benchmarks/chunk_memory.py /path/to/file 64 256 1024
```

### Implementation notes

- **SPSS/Stata chunks** are separate pyreadstat calls using `row_offset` and
  `row_limit`. Skipping to an offset is cheap. Reading the last 50k rows of a
  500k-row file costs 1.0× the first 50k for uncompressed SAV and DTA, and
  1.3–1.6× for compressed SAV/ZSAV. Larger chunks reduce this overhead further.
- **Arrow readahead.** Arrow's streaming CSV reader parses about 40 blocks ahead
  of the consumer. The reader therefore uses blocks of 8 MiB or less and joins
  them, without copying, into budget-sized chunks.
- **Stata `user_missing=True`.** Extended missings (`.a`–`.z`) come back as
  letters, so every Stata numeric column is `object` in every chunk. That costs
  about 4× the memory of float64, so enable it only when you need the codes.

## Options

All options are fields of `ReadOptions` and can also be passed to
`open_reader` as keyword arguments. See `src/metadata_tools/options.py`.

| Option | Default | |
|---|---|---|
| `usecols` | all | Columns to read (kept in file order) |
| `chunk_memory_bytes` | 256 MiB | Target in-memory size of a chunk |
| `chunk_rows` | None | Fixed rows per chunk instead of the memory budget |
| `max_full_read_bytes` | 2 GiB | Limit for `read()` |
| `dtype_backend` | `pyarrow` | `pyarrow` or `numpy` |
| `encoding` / `encodings` | auto | Forced encoding / encodings to try in order |
| `user_missing` | False | Keep SPSS user missings / Stata `.a`–`.z` |
| `convert_dates` | True | Convert date formats to datetime64 |
| `delimiter` | sniffed | CSV delimiter |
| `csv_type_inference` | `scan` | `scan`, `sample` or `string` |
| `dtypes` | {} | CSV per-column Arrow type overrides |

## Development

```
uv sync
uv run pytest
```

One test uses a real Stata file (`Brazil-2025-full data.dta`) that contains
latin-1 bytes in a file declared as UTF-8. The file is not in this repo; set
`METADATA_TOOLS_BRAZIL_DTA` to its path, otherwise the test is skipped.
