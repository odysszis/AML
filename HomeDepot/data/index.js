var csv       = require('csv-parse');
var fs        = require('fs');

require('should');

var output = [];

var input = fs.createReadStream('attributes.csv');

// Create the parser
var parser = csv({ delimiter: ',' });

parser.on('readable', function() {
  while (record = parser.read()) {
   // output.push(record);
    console.log(record);
  }
});

// Catch parse error
parser.on('error', function(err) {
  console.log("read error =",err.message);
});

parser.on('finish', function() {
  console.log("finish");
});

input.pipe(parser);
//parser.end();
//input.pipe(parser);
