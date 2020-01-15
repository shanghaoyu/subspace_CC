echo "Creating ccm inputs for SNM runs"
# for lambda delta = 500 use
#   c_D: -8.19999999999999951D-01
#   c_E: -3.49999999999999978D-01

#for lambda_delta = 450 use
#   c_D: 7.90000000000000036D-01
#   c_E: 1.70000000000000012D-02

dens_arr=(0.05 0.075 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.165 0.17 0.18 0.19 0.20)
for ((k=0;k<14;k++))
do
ed -s ccm_in <<EOF
H
2d
1a
3, 450
.
4d
3a
0.0170000000000000012, 0.790000000000000036
.
6d
5a
28
.
8d
7a
snm, density
.
12d
11a
${dens_arr[k]},1,2
.
14d
13a
CCD(T)
.
16d
15a
F, 0
.
18d
17a
450, 3
.
w ccm_in
EOF
echo
#more ccm_in
densidx=${dens_arr[k]}
filename="LO_nmax4_snm_${densidx}_450"
#cp ccm_in /lustre/atlas/scratch/pschwar3/nph008/CC_calcs/SNM-calcs/$filenam
cp ccm_in $filename
done
