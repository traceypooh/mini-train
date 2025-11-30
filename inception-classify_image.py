# TRACEY MODS TO:
#    ~/dev/models/tutorials/image/imagenet/classify_image.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import glob


# xxxx  feed to scikit cosine difference pairwise distance metrics package
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import pairwise_distances


def cos(sess, a, b):
    # each a,b are 4d arrays of dimens: [1][1][1][2048]
    return pdist([a[0][0][0], b[0][0][0]], 'cosine')[0]

    # xxxx  "feed to scikit cosine difference pairwise distance metrics package"
    hmm = [a[0][0][0], b[0][0][0]]
    return pairwise_distances(hmm[0:1], hmm, metric='cosine')[0,1]

    #print(sess.run(a))
    #print(sess.run(b))
    return cosine_distances(a[0][0], b[0][0])[0][0]

    return cosine(a, b)


FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# https://stackoverflow.com/questions/43357732/how-to-calculate-the-cosine-similarity-between-two-tensors
def cosine_similarity(x, y, normalize):
    """Computes cosign similarity of two tensors"""

    # x = tf.constant(np.random.uniform(-1, 1, 10))
    # y = tf.constant(np.random.uniform(-1, 1, 10)])
    if normalize:
        x = tf.nn.l2_normalize(x, 0)
        y = tf.nn.l2_normalize(y, 0)

    s = tf.losses.cosine_distance(x, dim=0) # xxx
    #print("s ", s)
    ret = tf.Session().run(s)
    #print("ret ", ret)
    # "Of course, 1 - s is the cosine similarity!"
    sim = 1 - ret
    #print("sim ", sim)
    return sim


    a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
    b = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_b")
    if normalize:
        normalize_a = tf.nn.l2_normalize(a, 0)
        normalize_b = tf.nn.l2_normalize(b, 0)
    else:
        normalize_a = a
        normalize_b = b
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    sess = tf.Session()
    cos_sim = sess.run(cos_similarity, feed_dict={a:x, b:y})
    #print(cos_sim)
    return cos_sim


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    #   xxx save these (for 2 images) and compare ('cosine compare??') them (hamming dist?)
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    ret1 = sess.run(pool_tensor, {'DecodeJpeg/contents:0': image_data})
    #print(ret1)


    for image2 in glob.glob('/Users/tracey/d/mini-train/__masked/*/*png'):
        im2 = tf.gfile.FastGFile(image2, 'rb').read()
        ret2 = sess.run(pool_tensor, {'DecodeJpeg/contents:0': im2})
        print(cos(sess, ret1, ret2), "\t", image, "\t", image2)
    return

    sim  = cosine_similarity(ret1, ret2, False)
    simN = cosine_similarity(ret1, ret2, True)
    diff = tf.squared_difference(ret1, ret2)
    #print("-------diff:")
    #print(diff)
    #print(diff.eval())
    #tf.Print(diff, [diff])
    xxx = tf.reduce_sum(diff)
    print(sim, "\t", simN, "\t", xxx.eval(), "\t", image2)
    return


    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    print(predictions)
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  # print('Loading: ', filepath)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
