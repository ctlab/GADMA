Frequency spectrum file format
********************************

Each file begins with any number of comment lines starting with ``#``.
The first non-comment line contains ``P`` integers giving the dimensions of the FS array, where ``P`` is the number of represented populations represented.
For an FS representing data from ``4x4x2`` samples, this would be ``5x5x3``.
(Each dimension is one larger than the number of samples because the number of observations can range, for example, from 0 to 4 if there are 4 samples, for a total of 5 possibilities.)
On the same line, the string ``folded`` or ``unfolded`` denotes whether or not the stored FS is folded.

The actual data is stored in a single line listing all the FS elements separated by spaces, in the order ``fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1]...``.
This is followed by a single line giving the elements of the mask in the same order as the data, with ``1`` indicating masked and ``0`` indicating unmasked.
