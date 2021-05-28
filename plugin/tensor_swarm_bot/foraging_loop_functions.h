#ifndef FORAGING_LOOP_FUNCTIONS_H
#define FORAGING_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/core/utility/math/range.h>
#include <argos3/core/utility/math/rng.h>
#include <jsoncpp/json/json.h>
#include <footbot_foraging.h>
#include <ros/ros.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "ma_foraging/AIService.h"
#include "shared_queue.hpp"
#include "client.h"
#include <unordered_map>

//#define USE_ROS_THREAD      0
//#define USE_ROS_SAFE_QUEUE  1

using namespace argos;

class CForagingLoopFunctions : public CLoopFunctions {
public:
    CForagingLoopFunctions();
    virtual ~CForagingLoopFunctions() {}

    virtual void Init(TConfigurationNode& t_tree);
    virtual void Reset();
    virtual void Destroy();
    virtual CColor GetFloorColor(const CVector2& c_position_on_plane);
    virtual void PreStep();
    virtual void PostStep();
    virtual bool IsExperimentFinished();
    
    /// Callback for the ROS service.
    //bool ServiceFunction(const ma_foraging::AIServiceRequest & req,
    //                     ma_foraging::AIServiceResponse &resp);

private:
    Real m_fFoodSquareRadius;
    CVector2 m_cFoodPos, m_cNestPos;
    /* The foraging parameters */
    CFootBotForaging::SForagingParams m_sForagingParams;
    
    CFloorEntity* m_pcFloor;
    std::string m_strOutput;
    std::ofstream m_cOutput;
    
    ma_foraging::AIServiceRequest m_req_store;
    ma_foraging::AIServiceResponse m_resp_store;
    std::unordered_map<std::size_t, Json::Value> m_mapMsgTbls;  //20200408
    std::unordered_map<std::size_t, std::size_t> m_mapIndexes;
    
    std::string m_sHost; int m_nPort;
    Client m_sClient;
    bool m_bAcceptClient;
    uint m_episode_time;
    uint m_persistence_cnt;
    std::unordered_map<std::size_t, Real> m_accRewards;
};

#endif
