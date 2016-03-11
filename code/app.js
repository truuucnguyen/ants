var sys = require('sys')
var exec = require('child_process').exec;
var child;
// executes `pwd`
child = exec("/usr/local/MATLAB/R2016a/bin/matlab -nojvm < train_normal_nn.m", function (error, stdout, stderr) {
  sys.print('stdout: ' + stdout);
  sys.print('stderr: ' + stderr);
  if (error !== null) {
    console.log('exec error: ' + error);
  }
});

