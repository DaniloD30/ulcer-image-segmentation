import argparse

from loadbatches import segmentGenerator
from models import vggsegnet

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--train_images", type=str)
parser.add_argument("--train_annotations", type=str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)

parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type=str, default="")
parser.add_argument("--val_annotations", type=str, default="")

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--val_batch_size", type=int, default=2)
parser.add_argument("--load_weights", type=str, default="")

parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--optimizer_name", type=str, default="adadelta")

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

m = vggsegnet.VGGSegnet(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy', \
          optimizer= optimizer_name, \
          metrics=['accuracy'])

if len(load_weights) > 0:
    m.load_weights(load_weights)
    
print("Model output shape", m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G = segmentGenerator(train_images_path, \
                     train_segs_path, \
                     train_batch_size, \
                     n_classes, \
                     input_height, \
                     input_width, \
                     output_height, \
                     output_width)

if validate:
    G2 = segmentGenerator(val_images_path, \
                          val_segs_path, \
                          val_batch_size, \
                          n_classes, \
                          input_height, \
                          input_width, \
                          output_height, \
                          output_width)
    
    for ep in range(epochs):
        m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))

else:
    for ep in range(epochs):
        m.fit_generator(G, 512, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
