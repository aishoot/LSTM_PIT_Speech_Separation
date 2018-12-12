## Reference

* [sphfile](https://github.com/mcfletch/sphfile)

## Usage
```python
from sphfile import SPHFile

sph =SPHFile('TEDLIUM_release2/test/sph/JamesCameron_2010.sph')
# Note that the following loads the whole file into ram
print( sph.format )
# write out a wav file with content from 111.29 to 123.57 seconds
sph.write_wav( 'test.wav', 111.29, 123.57 )
```
