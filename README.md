
###################################################################################################
# TO DO
- VR: get camera imagery (use as many frames as time allows -- repeated matches w phash?!)

# openface local install (xenial) - ssh io
```bash
# http://cmusatyalab.github.io/openface/setup/
# (building by hand)

############ setup
set -e
mkdir -m777 -p /var/cache/petabox/openface
ln -s          /var/cache/petabox/openface  /openface

apt-get -y install  python  python-setuptools  python-pip
python2 -m pip install --upgrade pip

############# install boost
apt-get -y install libboost-python-dev

python2 -m pip install numpy pandas scipy scikit-learn scikit-image


############# install dlib
cd /openface
apt-get -y install cmake
wget https://pypi.python.org/packages/e2/79/6aba1d2b3f9fbcf34d583188d8ff6818952ea875dceedf7c34a869637573/dlib-19.7.0.tar.gz
tar xf dlib*.tar.gz
cd dlib*/python_examples
mkdir -p build  &&  cd build  &&  \
  cmake ../../tools/python  &&  \
  cmake --build . --config Release  && \
  sudo cp dlib.so /usr/local/lib/python2.7/dist-packages

############# install openCV
cd /openface
apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev unzip
wget https://github.com/Itseez/opencv/archive/2.4.11.zip
unzip -q 2.4.11
cd opencv*/
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install

# verify above
python2 -c 'import cv2; import dlib;'

############# install torch
cd /openface
git clone https://github.com/torch/distro.git torch --recursive
cd torch  &&  bash install-deps  &&  cd -

# resolve libpng issues -- needs libpng16! -- fixes
BAD=/usr/lib/x86_64-linux-gnu/libpng.so
GOOD=/usr/local/lib/libpng.so
ls -l ${BAD?} ${GOOD?}
apt-get -y install zlib1g-dev
apt-get source libpng16-dev
rm -fv libpng*  |cat
cd libpng*1.6*/  &&  ./configure --prefix=/usr/local  &&  make -j4
sudo make install
cd /openface/torch
# NOTE: done twice so we can find the places to update libpng12 to libpng14
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./install.sh
perl -i -pe "s=${BAD?}=${GOOD?}="  $(grep -r ${BAD?} . |cut -f1 -d: |sort -u)
./install.sh
. /openface/torch/install/bin/torch-activate
# 'th' should work now!
for i in dpnn nn optim csvigo; do  luarocks  install  $i; done


############# setup main 'openface' repo
cd /openface
git clone https://github.com/cmusatyalab/openface
cd openface  &&  git submodule init  &&  git submodule update
python2 setup.py install
models/get-models.sh

############# patch code no longer compatible with python libs
cat << EOF | patch -p1
diff --git a/demos/classifier.py b/demos/classifier.py
index 8d67d09..b668e3b 100755
--- a/demos/classifier.py
+++ b/demos/classifier.py
@@ -43,2 +43,2 @@ from sklearn.svm import SVC
-from sklearn.grid_search import GridSearchCV
-from sklearn.mixture import GMM
+from sklearn.model_selection import GridSearchCV
+from sklearn.mixture import GaussianMixture as GMM
@@ -192 +192 @@ def infer(args, multiple=False):
-            person = le.inverse_transform(maxI)
+            person = le.inverse_transform([maxI])
@@ -200 +200 @@ def infer(args, multiple=False):
-                print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
+                print("Predict {} with {:.2f} confidence.".format(person, confidence))
EOF


# this should now work!
cd /openface  &&  . /openface/torch/install/bin/torch-activate  &&   /openface/openface/demos/classifier.py infer /home/tracey/d/mini-train/__features/classifier.pkl  /home/tracey/d/mini-train/jason-scott/JasonScott.jpg
```


###################################################################################################
# Ideas / Tips / Possible future research

### 24bit PNG 4-channel alpha (!!)
convert a.png -channel A -alpha background png24:a_png24_alpha.png

### imagemagick simpleton compare MAE
compare -verbose -metric MAE p1 p2 null:

- https://github.com/traceypooh/Facial-Similarity-with-Siamese-Networks-in-Pytorch
- https://github.com/FaceAR/OpenFaceIOS
- http://machinethink.net/blog/tensorflow-on-ios/
- https://www.tensorflow.org/tutorials/image_recognition
- https://www.tensorflow.org/install/install_mac
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios/
- https://developer.apple.com/documentation/arkit/creating_face_based_ar_experiences
- https://github.com/aleph7/AIImageCompare
# FLOWER RETRAINER -- RIK (from jwplayer) IDEA!
https://www.tensorflow.org/tutorials/image_retraining



###################################################################################################
# openface / facenet
# http://cmusatyalab.github.io/openface/demo-3-classifier/
```bash
# -crop WIDTHxHEIGHT+LEFT+TOP
# makes 1512x.. merged;  then crops to 32px WxH multiples
# cropping dimens for 'masked2' make result image 1120x1824 -- perfect 32px multiples for DCT phash..
# (run on home for md5-checking)
# NOTE: MANUALLY CROPPED TRAINING UNALIGNABLE DavidK/b.png
# NOTE: MANUALLY CROPPED TRAINING UNALIGNABLE DeniseB/IMG*
function openface-cropping() {
  cd ~/d/mini-train/pix;
  for FIN in $(ls */*.jpg); do
    NAME=$(dirname $FIN);
    PNG=$(echo $FIN |perl -pe 's/\.jpg/.png/i')
    pass=$(basename $FIN |cut -b1);
    out=../__masked/$PNG;
    mkdir -p $(dirname $out);

    CROP=1120x1824+186+184;
    typeset -a ARGS;
    ARGS="";
    if [ "$pass" = "b" ]; then
      CROP=1120x1824+160+180;
    elif [ "$pass" = "c" ]; then
      CROP=1120x1824+186+215;
      ARGS+=(-crop 1512x2016+0+300 +repage);
    elif [ "$pass" = "f" ]; then
      ARGS+=(-crop 1512x2016+25+0 +repage);
    fi;
    set -x;
    convert $FIN -auto-orient -resize 1512x $ARGS ../mask-black2.png -composite -crop $CROP +repage .tmp.png;
    { set +x; } 2>/dev/null
    a=$(cat $out 2>/dev/null |md5sum);
    b=$(cat .tmp.png         |md5sum);
    if [ "$a" != "$b" ]; then
      set -x;
      mv .tmp.png $out;
      { set +x; } 2>/dev/null
    fi
  done
  chmod -R ugo+rX ../__masked;
}; openface-cropping;
```

docker pull bamos/openface
docker run -v /home/tracey/d/mini-train:/root/openface/t -t -i bamos/openface /bin/bash
```bash
cd /root/openface
lt() { ls "$@" -altrh; }
find t/__aligned t/__features -type f -ls -delete
util/align-dlib.py t/__masked align outerEyesAndNose t/__aligned
identify t/__aligned/*/*.*

# OPTIONAL
# util/prune-dataset.py t/__aligned --numImagesThreshold 3;

# make .pkl
rm -fv t/__aligned/cache.t7  t/__features/*
batch-represent/main.lua -data t/__aligned -outDir t/__features
demos/classifier.py train t/__features


# IDENTIFY -- (with 20+ statues _not_ in mix, got) 13 of 13 right!!!
# later got 10 and 11 of 13 right (wrong: aaron/jim/ted; aaron/jim)
# LATER got 13 of 13 right!!
demos/classifier.py --verbose infer t/__features/*pkl t/__test/*.*
```



###################################################################################################
# Inception GoogLeNet -- use tensor vectors _instead_ before the classify like 'raccoon with confidence 0.3'
# step and just shortcut once vectors are made and run 'cosine distance' on them
```bash
# get the 2 relevant repos cloned to:
#   ~/dev/models
#   ~/dev/tensorflow
cd ~/dev/tensorflow
sudo easy_install pip
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages -p python3 .
source bin/activate

cd ~/dev/models/tutorials/image/imagenet
python3 ~/d/mini-train/inception-classify_image.py --image_file ~/d/mini-train/__masked/BrewsterKahle/a.png
```


##################################################################################################
# masking
# old ifone
- printed mask2-ants.png ("red ants") at 10%
  - cut out
  - taped to phone (white sides cut to end like _one extra_ red dash at bottom)
  - this guide bottom white sends then _started_ at bottom of _camera open_
```bash
cd ~/d/mini-train/pix;
convert ../mask-black2.png -resize 1224x1632 ../mask-half.png
convert ../mask-black2.png -resize 189x ../mask-black2small.png;
convert ../mask-black2.png -resize 378x ../mask-black4.png;
convert ../mask-black2.png -alpha extract -negate ../mask2.png;
convert ../mask-black4.png -alpha extract -negate ../mask4.png;


# -crop WIDTHxHEIGHT+LEFT+TOP
# makes 1224x1632 merged;  then crops to 32px WxH multiples
# cropping dimens for 'half' make the result image 896x1472 -- perfect 32px multiples for DCT phash..
for i in $(ls *JPG|mirror|cut -f2- -d.|mirror);do convert $i.JPG -auto-orient -resize 1224x ../mask-half.png -composite -crop 896x1472+161+150 +repage half/$i.png;done

for i in $(ls *JPG|mirror|cut -f2- -d.|mirror);do convert $i.JPG -auto-orient -resize 378x  ../mask-black4.png      -composite -crop 282x+47+47    +repage small4/$i.png;done
for i in $(ls *JPG|mirror|cut -f2- -d.|mirror);do convert $i.JPG -auto-orient -resize 378x  ../mask-black4.png      -composite ../mask4.png -compose CopyOpacity -composite -crop 282x+47+47 +repage small4/$i.png;done
for i in $(ls *JPG|mirror|cut -f2- -d.|mirror);do convert $i.JPG -auto-orient -resize 189x  ../mask-black2small.png -composite -crop 142x+23+23    +repage small/$i.png;done
```




###################################################################################################
# phash (better than imagehash -- but cImg based non-iOS was better!)
```bash
cd /Users/tracey/d/mini-train/pix/small;  f1=BrewsterKahle.png;        for f in *png; do ~/dev/phash/wtf $f1 $f; done |sort -nr
cd /Users/tracey/d/mini-train/pix/masked; f1=../BrewsterKahle-xxx.jpg; for f in *png; do ~/dev/phash/wtf $f1 $f; done |sort -nr

# phash1
brew install carthage
https://github.com/ameingast/cocoaimagehashing

# phash2
wget https://phash.org/releases/pHash-0.9.6.tar.gz
brew install cimg
# ..
./configure -disable-video_hash -disable-audio_hash -enable-image_hash  &&  make
```

###################################################################################################
# imagehash (distance hash -- fast but not high accuracy)
```bash
cd ~/dev/imagehash; php -r '$DIR="/Users/tracey/d/mini-train/pix/small"; $EXT="png"; require("ImageHash.php"); $f1="$DIR/BrewsterKahle.$EXT"; $h1=ImageHash::hash($f1); foreach(glob("$DIR/*.$EXT") as $f2){ $h2=ImageHash::hash($f2); echo ImageHash::distance($h1,$h2)."\t$f2\n"; }'|sort -nr
```
