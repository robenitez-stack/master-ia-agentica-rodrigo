# Reglas de trabajo del Tema 2

- Los notebooks `1_supervised_classification_model.ipynb` y `2_unsupervised_classification_model.ipynb` constituyen el módulo 1 histórico y no deben modificarse, moverse ni renombrarse.
- `input_notebooks/` contiene los ocho originales inmutables. Sus hashes deben coincidir con `SHA256SUMS.txt`.
- El desarrollo se realiza exclusivamente sobre copias canónicas de los módulos 2 y 3.
- Los módulos 4 y 5 permanecen pendientes hasta recibir sus materiales.
- Los datasets se almacenan en `.data/` y no se versionan.
- Los pesos y checkpoints no se versionan.
- Los resultados `fast` no son publicables ni pueden sustituir resultados `full`.
- Test se utiliza una sola vez al final; validation controla selección, checkpoint, scheduler y early stopping.
- Antes de cada commit se validan notebooks, hashes, artefactos, tamaños y estado Git.
- No se realiza push sin autorización expresa.

