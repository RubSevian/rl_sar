#ifndef RL_REAL_HPP
#define RL_REAL_HPP

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "unitree_legged_sdk/unitree_joystick.h"
#include <boost/bind.hpp>
#include <CustomInterface.h>
#include "loop.hpp"
// #include "unitree_legged_sdk/unitree_joystick.h"
#include <csignal>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using CyberdogData = Robot_Data;
using CyberdogCmd = Motor_Cmd;


class RL_Real : public RL ,public CustomInterface
{
public:
    RL_Real();
    ~RL_Real();
private:
// rl functions
    torch::Tensor Forward() override;
    // torch::Tensor ComputeObservation() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();

    // history buffer
    ObservationBuffer history_obs_buf;
    torch::Tensor history_obs;

    CyberdogData cyberdogData;
	CyberdogCmd cyberdogCmd;
    void UserCode();
	
    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;    
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;


   // float getup_percent = 0.0;
    //float getdown_percent = 0.0;
    //float start_pos[12];
	//float now_pos[12];

    // int control_state = STATE_WAITING;


    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // unitree interface
    //keyboard
    Control1 control;
    

    // others
    int motiontime = 0;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;
    int command_mapping[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
    int state_mapping[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
};

#endif