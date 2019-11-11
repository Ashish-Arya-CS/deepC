from mnist import MNIST
from subprocess import PIPE, run
import random, decimal
from PIL import Image
import numpy as np

mndata = MNIST('/home/amd/yann.lecun.mnist')

images, labels = mndata.load_testing()

# Write image tensor
def write_image(index):
  with open("image.data", "w") as fp:
    int_ary = [int(num) for num in images[index]]
    np_flt_ary  = np.array(int_ary, dtype=np.float)/255.0
    img = Image.fromarray(np_flt_ary.reshape(28,28), mode='L')
    fp.write(np.array_str(np_flt_ary).strip("[]"))

def run_model(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    for line in result.stdout.split("\n"):
      if ( line.find("writing file ") == 0 ):
        resultFile = line[13:line.find('.',-1)].split()[0]
        with open(resultFile, 'r') as f:
          return f.read()
    return ""

# Run model in the loop
import deepC.dnnc as dc
for i in range (5):
  index = random.randint(0,len(images)-1)
  write_image(index);

  model_result = run_model("./mnist.exe ./image.data").strip("[]")

  # Convert log softmax output to probability
  log_probs = dc.array([float(f) for f in model_result.strip("[]").split()])
  probabilities = dc.exp(log_probs)

  trueLabel  = labels[index]
  prediction = dc.argmax(probabilities)[0]
  print(mndata.display(images[index]))
  print("True label = ", labels[index])
  print("Model Prediction: ", dc.argmax(probabilities))