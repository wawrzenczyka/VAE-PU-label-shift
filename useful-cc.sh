cd VAE-PU-label-shift-CC/
conda activate vae-pu-env
rm -rf result-clean/
rm -rf result-clean-CC/
python ./create_results_copy.py
mv result-clean/ result-clean-CC/
tar -czf results.tar.gz result-clean/
cp results.tar.gz results-2024-06-11.tar.gz

scp awawrzenczyk@eden.mini.pw.edu.pl:~/VAE-PU-label-shift-CC/results.tar.gz \
results-CC-2024-06-11.tar.gz

scp wawrzenczyka@ssh.mini.pw.edu.pl:~/results-CC-2024-06-11.tar.gz results-CC-2024-06-11.tar.gz
