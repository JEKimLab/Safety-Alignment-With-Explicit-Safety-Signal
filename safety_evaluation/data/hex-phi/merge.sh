> all.csv

for file in *.csv; do
    tail -n +1 "$file" >> all.csv
done
