# Instruction on Passwordlessly Connecting to Cluster by Setting Up SSH Key Authentification

## Preface
This instruction is for students who suffer from typing the complicated password every time to login the cluster. We provide a step-by-step guide to help you set up SSH key authentification for passwordless connection. After executing the commands following this instruciton, you should be able to login the cluster without typing password.

## Contact
If you encounter any problem following this instruction, please first have a look at the FAQ part and ask help if nothing helps:

**Name:** Liu Yuxuan

**Email:** [118010200@link.cuhk.edu.cn](mailto:118010200@link.cuhk.edu.cn)

**Position:** Teaching Assistant

## 1. Generate SSH Key Pair on Your Local Machine
```bash
## Execute the following commands on your local machine
cd ~/.ssh
ssh-keygen -t rsa -b 1024 -f csc4005_rsa -C "{Put Any Comment You Like}"
ssh-add -K ./csc4005_rsa
```
To check if you make it right, type the following command and you should see a string as the output.
```bash
cat ~/.ssh/csc4005_rsa.pub
```
**Notes:**
- Execute ssh-keygen on your own computer instead of the cluster.
- It is fine to use the SSH key file generated before (eg. id_rsa.pub), if you have.
- ssh-keygen will ask you to set a paraphrase, which improves the security of using SSH key authentification. Type "Enter" for no paraphrase.

**Parameters for ssh-keygen command:**
- **-t**: type for ssh key generation. Here we use rsa
- **-b**: bits
- **-f**: name of your ssh key file. You are recommended to set this parameter in case it
is conflict with the ssh key file you generated before.
- **-C**: comments to distinguish the ssh key file from others

## 2. Transfer Your SSH Public Key File to the Cluster
```bash
## Execute the following commands on your local machine
scp ~/.ssh/csc4005_rsa.pub {Your Student ID}@10.26.200.21:~
```
**Notes:**
- `{Your Student ID}` is where you put your student ID. Do not add the {} brackets in your command.
- This command will ask you for password for data transfer. Please make sure that you type in the correct password. If you are Windows user and you want to copy & paste the password for convenience, try "Ctrl + Shift + V" if you fail to paste the password with "Ctrl + V".

## 3. Configure authorized_keys on the Cluster
```bash
## Login the cluster with password
ssh {Your Student ID}@10.26.200.21
## All the following commands should be executed on the cluster
mkdir -p ~/.ssh
cat ./csc4005_rsa.pub >> ~/.ssh/authorized_keys
rm -f ~/csc4005_rsa.pub
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```
To check if you make it right, execute the following command and you should see a string as the output that is the same as in Step-1.
```bash
cat ~/.ssh/authorized_keys
```

## 4. Prepare for SSH Connection with Key Authentification
Add the following content to ~/.ssh/config file on your local machine:
```bash
Host CSC4005_cluster
    HostName 10.26.200.21
    IdentityFile ~/.ssh/csc4005_rsa
    User {Your Student ID}
```
**Notes:**
- It is recommended to configure with the RemoteSSH extension on VSCode. Please refer to [Remote Development using SSH](https://code.visualstudio.com/docs/remote/ssh) for more infomation.

# 5. Login the Cluster Passwordlessly with SSH key Authentification
```bash
ssh {Your Student ID}@10.26.200.21
```
or
```bash
ssh {Your Student ID}@CSC4005_cluster
```

## FAQ
### 1. Why the password is still required to login the cluster after executing all the commands correctly in this instruction?

**Answer:** Execute the following command and you will see detailed debug logs for your SSH connection. Check the logs and see if you could find out what is going wrong and fix it. If you are Mac user, you can also refer to this article [Troubleshooting passwordless login](https://help.dreamhost.com/hc/en-us/articles/215906508-Troubleshooting-passwordless-login) to see if it could help you. If nothing helps, take a screenshot of the output of the command, and send it to [118010200@link.cuhk.edu.cn](mailto:118010200@link.cuhk.edu.cn).
```bash
ssh -vvv {Your Student ID}@10.26.200.21
```

## Appendix

### How does SSH private/public key pair guarantees security of your connection?
- **Step-1:** The client begins by sending an ID for the key pair it would like to authenticate with to the server.
- **Step-2:** The server check's the authorized_keys file of the account that the client is attempting to log into for the key ID.
- **Step-3:** If a public key with matching ID is found in the file, the server generates a random number and uses the public key to encrypt the number.
- **Step-4:** The server sends the client this encrypted message.
- **Step-5:** If the client actually has the associated private key, it will be able to decrypt the message using that key, revealing the original number.
- **Step-6:** The client combines the decrypted number with the shared session key that is being used to encrypt the communication, and calculates the MD5 hash of this value.
- **Step-7:** The client then sends this MD5 hash back to the server as an answer to the encrypted number message.
- **Step-8:** The server uses the same shared session key and the original number that it sent to the client to calculate the MD5 value on its own. It compares its own calculation to the one that the client sent back. If these two values match, it proves that the client was in possession of the private key and the client is authenticated.

For more information, please refer to following articles:

[Basic overview of SSH Keys](https://www.ssh.com/academy/ssh-keys)

[Understanding the SSH Encryption and Connection Process](https://www.digitalocean.com/community/tutorials/understanding-the-ssh-encryption-and-connection-process)
