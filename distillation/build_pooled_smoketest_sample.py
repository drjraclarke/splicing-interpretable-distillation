import pandas as pd

DATA_DIR = "/nfs/users/nfs_j/jc58/repos/splicing-interpretable-distillation/distillation/data"

train = pd.read_csv(f"{DATA_DIR}/fas_pooled16_hek_train.csv.gz")
test = pd.read_csv(f"{DATA_DIR}/fas_pooled16_hek_test.csv.gz")

train.sample(n=5000, random_state=0).to_csv(f"{DATA_DIR}/fas_pooled16_hek_smoketest_train.csv.gz", index=False)
test.sample(n=1000, random_state=0).to_csv(f"{DATA_DIR}/fas_pooled16_hek_smoketest_test.csv.gz", index=False)
print("done")
