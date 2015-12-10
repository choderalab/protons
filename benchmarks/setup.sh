#!/usr/bin/env zsh

. /opt/amber/amber.sh
start=`pwd`

# Run tleap on the newly generated input files
tleap -f tleap.in >! tleap.log
  
# There might be other error clues. This method isn't fail safe.
tleap_result=$(grep "usage" tleap.log || grep -i "error" tleap.log)

# As long as the grep results are empty
if [ -z "$tleap_result" ]
then
  echo "Tleap looks successful."
else
  echo "Caught an error in Tleap. Tough luck, buddy."
fi
  
cpinutil.py -resnames HIP GL4 AS4 -p complex.prmtop -o complex.cpin
cd ${start}
done
