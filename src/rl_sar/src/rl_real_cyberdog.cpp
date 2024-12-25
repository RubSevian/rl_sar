#include "../include/rl_real_cyberdog.hpp"

// #define PLOT
 // #define CSV_LOGGER

RL_Real rl_sar;

RL_Real::RL_Real() : CustomInterface(500)
{
    //torch::autograd::GradMode::set_enabled(false);
    // read params from yaml
    this->robot_name = "a1_isaacgym"; //"Cyberdog";
    this->ReadYaml(this->robot_name);
    for (std::string &observation : this->params.observations)
    {
        if (observation == "ang_vel"){
            observation = "ang_vel_body";
        }
    }
    
    //init rl
    torch::autograd::GradMode::set_enabled(false);
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size());
    }


    // history
    //this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, 6);

    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    running_state = STATE_WAITING;

    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control  = std::make_shared<LoopFunc>("loop_control", 0.002, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl",  this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
    //this->loop_rl->start();
    

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for(auto& vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for(auto& vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot    = std::make_shared<LoopFunc>("loop_plot"   , 0.002,    std::bind(&RL_Real::Plot,         this));
    this->loop_plot->start();
#endif

#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

//void RL::KeyboardInterface(){}


void RL_Real::GetState(RobotState<double> *state)
{

    // if((int)this->unitree_joy.btn.components.R2 == 1)
    // {keyboard
    //     this->control.control_state = STATE_POS_GETUP;
    // }
    // else if((int)this->unitree_joy.btn.components.R1 == 1)
    // {
    //     this->control.control_state = STATE_RL_INIT;
    // }
    // else if((int)this->unitree_joy.btn.components.L2 == 1)
    // {
    //     this->control.control_state = STATE_POS_GETDOWN;
    // }

    //-------NEED_TEST---------//

    if (this->params.framework == "isaacgym"){
        state->imu.quaternion[3] = this->cyberdogData.quat[1]; // w
        state->imu.quaternion[0] = this->cyberdogData.quat[2]; // x
        state->imu.quaternion[1] = this->cyberdogData.quat[3]; // y
        state->imu.quaternion[2] = this->cyberdogData.quat[0]; // z
    }
    else if (this->params.framework == "isaacsim")
    {
        state->imu.quaternion[0] = this->cyberdogData.quat[1]; // w
        state->imu.quaternion[1] = this->cyberdogData.quat[2]; // x
        state->imu.quaternion[2] = this->cyberdogData.quat[3]; // y
        state->imu.quaternion[3] = this->cyberdogData.quat[0]; // z
    }
    for(int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->cyberdogData.omega[i];
    }
    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->cyberdogData.q[state_mapping[i]];
        state->motor_state.dq[i] = this->cyberdogData.qd[state_mapping[i]];
        state->motor_state.tauEst[i] = this->cyberdogData.tau[state_mapping[i]];
    }
}

void RL_Real::UserCode()
{
	cyberdogData = robot_data;
	motor_cmd = cyberdogCmd;
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        
        this->cyberdogCmd.q_des[i] = command->motor_command.q[command_mapping[i]];
        this->cyberdogCmd.qd_des[i] = command->motor_command.dq[command_mapping[i]];
        this->cyberdogCmd.kp_des[i] = command->motor_command.kp[command_mapping[i]];
        this->cyberdogCmd.kd_des[i] = command->motor_command.kd[command_mapping[i]];
        this->cyberdogCmd.tau_des[i] = command->motor_command.tau[command_mapping[i]];
    }


}


void RL_Real::RobotControl()
{
    this->motiontime++;

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if(this->running_state == STATE_RL_RUNNING)
    {
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        torch::Tensor clamped_actions = this->Forward();

        for (int i : this->params.hip_scale_reduction_indices)
        {
            clamped_actions[0][i] *= this->params.hip_scale_reduction;
        }

        this->obs.actions = clamped_actions;

        torch::Tensor origin_output_torques = this->ComputeTorques(this->obs.actions);

        this->TorqueProtect(origin_output_torques);

        this->output_torques = torch::clamp(origin_output_torques, -(this->params.torque_limits), this->params.torque_limits);
        this->output_dof_pos = this->ComputePosition(this->obs.actions);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tauEst).unsqueeze(0);
        this->CSVLogger(this->output_torques, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}


torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->cyberdogData.q[i]);
        this->plot_target_joint_pos[i].push_back(this->cyberdogCmd.q_des[i]);
        plt::subplot(4, 3, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

void signalHandler(int signum)
{
    system("ssh root@192.168.55.233 \"ps | grep -E 'manager|ctrl|imu_online' | grep -v grep | awk '{print \\$1}' | xargs kill -9\"");
    system("ssh root@192.168.55.233 \"export LD_LIBRARY_PATH=/mnt/UDISK/robot-software/build;/mnt/UDISK/manager /mnt/UDISK/ >> /mnt/UDISK/manager_log/manager.log 2>&1 &\"");

    exit(0);
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    while(1)
    {
        sleep(10);
    };

    return 0;
}