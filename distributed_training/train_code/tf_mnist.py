# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tensorflow as tf
import argparse
import os

def train(args):
    tf.random.set_seed(args.seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        # Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
        tf.config.experimental.set_visible_devices(gpus[args.local_rank], 'GPU')

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % args.rank)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                 tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()

    opt = tf.optimizers.Adam(0.0001*args.world_size)

    checkpoint_dir = args.model_dir
    checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

    for batch, (images, labels) in enumerate(dataset.take(10000 // args.world_size)):
        loss_value = training_step(mnist_model, loss, opt, images, labels)

        if batch % 50 == 0 and args.rank == 0:
            print('Step #%d, \tLoss: %.6f,' % (batch, loss_value))

    # Save checkpoints only from master node.
    if args.rank == 0:
        mnist_model.save(os.path.join(checkpoint_dir, '1'))


@tf.function
def training_step(mnist_model, loss, opt, images, labels):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    return loss_value
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Setting for Distributed Training
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)


    # SageMaker Container environment
    parser.add_argument('--model-dir', type=str, default='../model')
    parser.add_argument('--data-dir', type=str, default='../data')
    
    args = parser.parse_args()
    
    try:
        args.model_dir = os.environ['SM_MODEL_DIR']
        args.data_dir = os.environ['SM_CHANNEL_TRAINING']
    except KeyError as e:
        print("The model starts training on the local host without SageMaker TrainingJob.")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        pass

    train(args)    