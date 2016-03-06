var csv = require('csv-parse');
var fs = require('fs');

require('should');

var input = fs.createReadStream('product_descriptions.csv');

// Create the parser
var parser = csv({
  delimiter: ','
});

parser.on('readable', function() {
  while (record = parser.read()) {
    // output.push(record);¬
    if (!record[0] || record[0] === 'product_uid') console.log(record);
    else {
      fs.writeFile("./corpus/" + record[0] + ".txt", record[1], function(err) {
        if (err) {
          return console.log(err);
        }
        console.log("file saved");
      });
    }
  }
});

// Catch parse error
parser.on('error', function(err) {
  console.log("read error =", err.message);
});

parser.on('finish', function() {
  console.log("finish");
  parser.end();
});

input.pipe(parser);

