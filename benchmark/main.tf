provider "aws" {
  region = "us-west-2"
}

variable "availability_zone" {
  type = string
  default = "us-west-2b"
}

variable "docker_username" {
  type = string
}

variable "docker_password" {
  type = string
}

resource "aws_instance" "master" {
  ami = "ami-089e3eddcb00a64ee"
  instance_type = "g4dn.2xlarge"
  availability_zone = var.availability_zone
  subnet_id = var.availability_zone == "us-west-2a" ? "subnet-0beb74857e3668908" : "subnet-02e3ebfa27314e526"
  placement_group = "pollux"
  root_block_device {
    volume_size = "256"
  }
  tags = {
    Name = "pollux-master"
    Team = "Pollux"
    custodian-opt-out-stop-nightly = "off"
  }
  security_groups = ["sg-06e00935bdb4c0ee4", "sg-048308a6c6238a567"]
  key_name = "esper"

  connection {
    type = "ssh"
    user = "ubuntu"
    host = self.private_ip
    private_key = file("~/.ssh/esper.pem")
    agent = true
  }

  provisioner "remote-exec" {
    inline = [
      "sudo snap install yq --channel=v3/stable",
      "sudo kubeadm init --pod-network-cidr=192.168.0.0/16",
      "mkdir -p ~/.kube",
      "sudo cp /etc/kubernetes/admin.conf ~/.kube/config",
      "sudo chown ubuntu:ubuntu ~/.kube/config",
      "kubectl apply -f https://docs.projectcalico.org/v3.11/manifests/calico.yaml",
      "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta5/nvidia-device-plugin.yml",
      "kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/common.yaml",
      "kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/operator.yaml",
      join(" ", ["curl -s https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/cluster.yaml",
                 "| /snap/bin/yq w - spec.storage.deviceFilter nvme0n1p2 | kubectl apply -f -"]),  # Use nvme0n1p2 for CephFS.
      "kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/filesystem.yaml",
      "kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/csi/cephfs/storageclass.yaml",
      "docker login -u ${var.docker_username} -p '${var.docker_password}'",
      join(" ", ["kubectl create secret generic regcred",
                 "--from-file=.dockerconfigjson=/home/ubuntu/.docker/config.json",
                 "--type=kubernetes.io/dockerconfigjson"]),
      "helm repo add stable https://charts.helm.sh/stable --force-update",
      "mkdir ~/pollux",
    ]
  }

  provisioner "file" {
    source = "../"
    destination = "~/pollux"
  }

  provisioner "remote-exec" {
    inline = [
      "kubectl create -f ~/pollux/benchmark/datasets.yaml",
      "kubectl create -f ~/pollux/benchmark/pvc.cephfs.yaml",
      "conda env update -f ~/pollux/benchmark/environment.yaml",
    ]
  }
}

resource "aws_instance" "workers" {
  depends_on = [aws_instance.master] 
  count = "16"
  ami = "ami-089e3eddcb00a64ee"
  instance_type = "g4dn.12xlarge"
  availability_zone = var.availability_zone
  subnet_id = var.availability_zone == "us-west-2a" ? "subnet-0beb74857e3668908" : "subnet-02e3ebfa27314e526"
  placement_group = "pollux"
  root_block_device {
    volume_size = "256"
  }
  tags = {
    Name = "pollux-${count.index}"
    Team = "Pollux"
    custodian-opt-out-stop-nightly = "off"
  }
  security_groups = ["sg-06e00935bdb4c0ee4", "sg-048308a6c6238a567"]
  key_name = "esper"

  connection {
    type = "ssh"
    user = "ubuntu"
    host = self.private_ip
    private_key = file("~/.ssh/esper.pem")
    agent = true
  }

  provisioner "remote-exec" {
    inline = [
      # Create two partitions: nvme0n1p1 for datasets, and nvme0n1p2 for CephFS.
      "printf ',500G\n;\n' | sudo sfdisk /dev/nvme0n1",
      "sleep 5",
      "sudo mkfs.ext4 -E nodiscard /dev/nvme0n1p1",
      "sudo mount /dev/nvme0n1p1 /mnt",
      "sudo chmod 777 /mnt",
      "docker login -u ${var.docker_username} -p '${var.docker_password}'",
      join(" ", ["sudo $(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                 "${aws_instance.master.private_ip} kubeadm token create --print-join-command)"]),
    ]
  }

  provisioner "file" {
    source = "~/.aws"
    destination = "~/.aws"
  }
}

output "master" {
  value = aws_instance.master.private_ip
}
