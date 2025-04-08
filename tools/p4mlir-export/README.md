CMD TO RUN P4HIRTOJSON:
```
{bin}/p4mlir-export --json-output {p4mlir-export}/bool_test.mlir > {p4mlir-export}/bool_test.json
```

CMD TO GET BMV2 JSON FROM P4
```
{bin}/p4c --target bmv2 --arch v1model -o simplebmv2.json simplebmv2.p4
```
