# DemHis input files' formats

## VCF data format

If you have .vcf file, you can convert it into .sfs file using [https://github.com/isaacovercast/easySFS].

## Frequency spectrum file format

Each file begins with any number of comment lines beginning with #. The first non-comment line contains P integers giving the dimensions of the FS array, where P is the number of populations represented. For a FS representing data from 4x4x2 samples, this would be 5 5 3. (Each dimension is one larger than the number of samples, because the number of observations can range, for example, from 0 to 4 if there are 4 samples, for a total of 5 possibilities.) On the same line, the string folded or unfolded denoting whether or not the stored FS is folded.

The actual data is stored in a single line listing all the FS elements separated by spaces, in the order fs[0,0,0] fs[0,0,1] fs[0,0,2]. . . fs[0,1,0] fs[0,1,1]. . . . This is followed by a single line giving the elements of the mask in the same order as the data, with 1 indicating masked and 0 indicating unmasked.

## SNP data format

The data file begins with any number of comment lines that being with #. The first parsed line is a column header line. Whitespace is used to separate entries within the table, so no spaces are allowed within any entry. Individual rows make be commented out using #.

The first column contains the in-group reference sequence at that SNP, including the flanking bases. If the flanking bases are unknown, they can be denoted by -. The header label is arbitrary.

The second column contains the aligned outgroup reference sequence at that SNP, including the flanking bases. Unknown entries can be denoted by -. The header label is arbitrary.
The third column gives the first segregating allele. The column header must be exactly Allele1.

Then follows an arbitrary number of columns, one for each population, each giving the number of times Allele1 was observed in that population. The header for each column should be the population identifier.

The next column gives the second segregating allele. The column header must be exactly Allele2.

Then follows one column for each population, each giving the number of times Allele2 was observed in that population. The header for each column should be the population identifier, and the columns should be in the same order as for the Allele1 entries.

Then follows an arbitrary number of columns which will be concatenated with _ to assign a label for each SNP.

The Allele1 and Allele2 headers must be exactly those values because the number of columns between those two is used to infer the number of populations in the file.

