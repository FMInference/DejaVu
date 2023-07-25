# Hack the CrusoeCloud Platform

## Setup a Coordinator

- So for, the CrusoeCloud Platform can only allow MacOS CLI tool. So we need to first setup a MacOS instance on AWS.
- Follow this link (https://medium.com/aws-architech/how-to-run-macos-using-amazon-ec2-mac-instances-cur-d918094f9b65) to setup the MacOS instance.
- Login into that node by SSH, username is 'ec2-user'.
- Install memcache on that instance:

      brew update
      brew upgrade
      brew install crusoecloud/cli/crusoe

- Configure crusoe conf file.
  - Make a ~/.crusoe/config file;
  - Put the following thing in it:
  
        [default]
        access_key_id="<access_key_id>"
        secret_key="<secret_key>"
- 