name=$1
node_type=$2
crusoe vm create --format json --name $name --type $node_type --keyfile ~/.ssh/id_ed25519.pub
#--startup-script ~/CrusoeHack/crusoe_scripts/startup_install.sh  #this does not work for some reason