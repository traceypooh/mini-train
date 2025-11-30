<?require 'ia';

// script for iOS 'miniARchive' app
// receives a processed camera image from the app and returns statue name that matches it best

// @see /home/tracey/d/mini-train/README.me  to setup openface and dependencies


$dir = '/tmp/ARchive';

if (!file_exists($dir))
  mkdir($dir, 0777, true);

$fi = tempnam($dir, 'shot');
// nosemgrep: php.lang.security.unlink-use.unlink-use system generated tmpfile
unlink($fi);
$fi .= '.png';
file_put_contents($fi, base64_decode($_POST['img']));


// Use (the incredible) OpenFace for matching
$ret = Util::cmd(
  '. /openface/torch/install/bin/torch-activate  &&  ' .
  '/openface/openface/demos/classifier.py infer /home/tracey/d/mini-train/__features/classifier.pkl ' .
  Util::esc($fi) . ' 2>&1 |tail -1',
  'CONTINUE'
);
error_log($ret);

list($check, $name,) = explode(' ', $ret);
if ($check == 'Predict') {
  $name = trim(preg_replace('/([A-Z])/', ' $1', $name));
  $name = trim(trim($name, "[]'"));
  echo $name;
  error_log("PREDICTED: $name");
  // nosemgrep: php.lang.security.unlink-use.unlink-use system generated tmpfile
  unlink($fi);
  exit;
}


// fallback to very crappy MAE pixel difference comparisons
$best = 1000;
$bestFI = '';
foreach (glob('/home/tracey/dev/cmp/*.png') as $img) {
  list(,,$cmp) = explode(' ', trim(Util::cmd('/usr/bin/compare -verbose -metric MAE ' . Util::esc($img) .
                         ' ' . Util::esc($fi) . ' null: 2>&1 |fgrep all:')));
  $cmp = trim($cmp, '()');
  if (!$bestFI  ||  $cmp < $best) {
    $best = $cmp;
    $bestFI = basename($img);
  }
  //echo basename($img) . " => $cmp\n";
}
// nosemgrep: php.lang.security.unlink-use.unlink-use system generated tmpfile
unlink($fi);

echo trim(preg_replace('/([A-Z])/', ' $1', preg_replace('/\.png$/', '', $bestFI)));
