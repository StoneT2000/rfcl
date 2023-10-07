# Demonstration Data

## ManiSkill 2

To use ManiSkill 2 demonstrations, we must download and then process/format them. The only processing done is to simply add in the observations to the demonstrations which are left out by default for space conservation.

```bash
# download demonstrations
python -m mani_skill2.utils.download_demo "PickCube-v0"
python -m mani_skill2.utils.download_demo "PegInsertionSide-v0"
python -m mani_skill2.utils.download_demo "StackCube-v0"
python -m mani_skill2.utils.download_demo "PlugCharger-v0"

# format demonstrations
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/v0/rigid_body/PickCube-v0/trajectory.h5 -c "pd_ee_delta_pose" -o state --num-procs 10 --save-traj
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5 -c "pd_ee_delta_pose" -o state --num-procs 10 --save-traj
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/v0/rigid_body/StackCube-v0/trajectory.h5 -c "pd_ee_delta_pose" -o state --num-procs 10 --save-traj
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/v0/rigid_body/PlugCharger-v0/trajectory.h5 -c "pd_ee_delta_pose" -o state --num-procs 10 --save-traj
```

## Adroit

The Adroit demonstrations are human tele-operated demonstrations. 

```bash
# download the original tele-operated demonstration data

# format the demonstrations into the ManiSkill2 format
python scripts/adroit/format_dataset.py
```
## Metaworld

The Metaworld demonstrations are generated via scripted policies provided by the original paper. To generate these run

```bash
python scripts/metaworld/format_dataset.py
```