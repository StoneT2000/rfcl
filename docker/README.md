# Docker Setup

First build the base docker image (includes conda/mamba and various other common packages)

```
docker build -t rfcl_base base 
```

To setup a docker image for ManiSkill2:

```
docker build -t rfcl_ms2 ms2 
```

To setup a docker image for Adroit:

```
docker build -t rfcl_adroit adroit 
```

To setup a docker image for Metaworld

```
docker build -t rfcl_metaworld meta-world
```