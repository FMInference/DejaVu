cd ~/GPT-home-private

if [ $# -eq 0 ]
then
  git pull
else
  token=$1
  git pull https://"${token}"@github.com/BinhangYuan/GPT-home-private.git
fi