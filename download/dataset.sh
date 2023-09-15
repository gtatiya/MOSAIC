
mkdir data
wget --no-check-certificate -O data/object_labels.zip https://tufts.box.com/shared/static/m6xdyiozpywxt5u1xsxlfoevljw90ocu.zip
unzip -o data/object_labels.zip -d data
rm data/object_labels.zip

wget --no-check-certificate -O data/cy101_Binary.tar.gz https://tufts.box.com/shared/static/rdf1j8tf83wv8stehym9fjrkn2a8aecw.gz
tar -xzvf data/cy101_Binary.tar.gz --one-top-level=data/
rm data/cy101_Binary.tar.gz
