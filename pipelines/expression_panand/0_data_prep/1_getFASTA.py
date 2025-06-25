import sys
sys.path.append('/workdir/jz963/Expression_modeling/a2z/a2z-expression/src/python/a2ze/src')
import a2ze.data
from a2ze.data import transforms
from tqdm import tqdm

def write_fasta(names, sequences, filename):
    with open(filename, 'w') as f:
        for name, sequence in zip(names, sequences):
            f.write(f">{name}\n")
            f.write(f"{sequence}\n")

trainDataset = a2ze.data.Dataset(path = 'a2z_dataset', task = sys.argv[1], split = sys.argv[2])

trainTransform = transforms.Compose([transforms.Flank(trainDataset, upstream=int(sys.argv[4])),
                                transforms.ExtractSequence(trainDataset)])

sequences = []
geneid = []
for i in tqdm(a2ze.data.TransformedDataset(trainDataset, trainTransform)):
    sequences.append(i['sequence'])
    geneid.append(i['id'])

write_fasta(names = geneid, sequences = sequences, filename = sys.argv[3])