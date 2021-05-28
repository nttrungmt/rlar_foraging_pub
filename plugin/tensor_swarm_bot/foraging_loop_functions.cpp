#include "foraging_loop_functions.h"
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <geometry_msgs/Pose2D.h>
#include <sys/file.h>
#include <errno.h>
#include <functional>
#include <iostream>
#include "ros_helpers.h"
#include <ctime>
#include <time.h>
#include <jsoncpp/json/writer.h>
#define NANO_SECOND_MULTIPLIER  1000000  // 1 millisecond = 1,000,000 Nanoseconds
const long INTERVAL_MS = 100 * NANO_SECOND_MULTIPLIER;

std::string getNodeName() {
    int i = 0;
    std::string name;
    do {
        name = "ai" + std::to_string(i);
        const std::string filename = "/tmp/argos_" + name + ".pid";
        int pid_file = open(filename.c_str(), O_CREAT | O_RDWR, 0666);
        int rc = flock(pid_file, LOCK_EX | LOCK_NB);
        if(rc) {
            std::cout << "instance running: " << filename << std::endl;
            if(EWOULDBLOCK == errno)
                std::cout << "instance running: " << filename << std::endl;
        } else {
            break;
        }
        ++i;
    } while(true);
    
    return name;
}

/****************************************/
/****************************************/
CForagingLoopFunctions::CForagingLoopFunctions() :
        m_sHost("127.0.0.1"),
        m_nPort(12345),
        m_pcFloor(NULL),
        //m_service_data_available(false), 
        //m_loop_done(false), 
        m_bAcceptClient(false),
        m_episode_time(0),
        m_persistence_cnt (0) {
// #if defined(USE_ROS_THREAD) || defined(USE_ROS_SAFE_QUEUE)
    // int argc = 0;
    // char *argv = (char *) "";
    
    // std::string name = getNodeName();
    // std::cout << "Node name: " << name << std::endl;
    // ros::init(argc, &argv, name);
    
    // m_ros_thread = std::thread([this, name]() {
    // ros::NodeHandle n;
    // auto service =
            // n.advertiseService<ma_foraging::AIServiceRequest, ma_foraging::AIServiceResponse>
                    // (name,
                     // std::bind(
                             // &CForagingLoopFunctions::ServiceFunction,
                             // this,
                             // std::placeholders::_1,
                             // std::placeholders::_2));
    // ros::spin();
    // });
//#endif
}

/****************************************/
/****************************************/
void CForagingLoopFunctions::Init(TConfigurationNode& t_node) {
    try {
        TConfigurationNode& tForaging = GetNode(t_node, "foraging");
        // Get a pointer to the floor entity
        m_pcFloor = &GetSpace().GetFloorEntity();
        // Get the communication host
        GetNodeAttribute(tForaging, "host", m_sHost);
        // Get the communication port
        GetNodeAttribute(tForaging, "port", m_nPort);
        // Get the other foraging params
        m_sForagingParams.Init(tForaging);
        // Get the location of FOOD / NEST and their radius
        //GetNodeAttribute(tForaging, "radius", m_fFoodSquareRadius);
        m_fFoodSquareRadius = m_sForagingParams.m_fFoodSquareRadius;
        m_fFoodSquareRadius *= m_fFoodSquareRadius;
        //Real fFoodPosX, fFoodPosY, fNestPosX, fNestPosY;
        //GetNodeAttribute(tForaging, "foodPosX", fFoodPosX);
        //GetNodeAttribute(tForaging, "foodPosY", fFoodPosY);
        //GetNodeAttribute(tForaging, "nestPosX", fNestPosX);
        //GetNodeAttribute(tForaging, "nestPosY", fNestPosY);
        m_cFoodPos = CVector2(m_sForagingParams.m_fFoodPosX, m_sForagingParams.m_fFoodPosY);
        m_cNestPos = CVector2(m_sForagingParams.m_fNestPosX, m_sForagingParams.m_fNestPosY);
        // Get the output file name from XML
        GetNodeAttribute(tForaging, "output", m_strOutput);
        // Open the file, erasing its contents
        m_cOutput.open(m_strOutput.c_str(), std::ios_base::app | std::ios_base::out);
        m_cOutput << "# clock\twalking\tresting\tcollected_food\tenergy" << std::endl;
    } catch(CARGoSException& ex) {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }
}

/****************************************/
/****************************************/
void CForagingLoopFunctions::Reset() {
    /* Close the file */
    m_cOutput.close();
    /* Open the file, erasing its contents */
    m_cOutput.open(m_strOutput.c_str(), std::ios_base::trunc | std::ios_base::out);
    m_cOutput << "# clock\twalking\tresting\tcollected_food\tenergy" << std::endl;
    m_episode_time = 0;
    m_persistence_cnt = 0;
    m_accRewards.clear();
}

/****************************************/
/****************************************/
void CForagingLoopFunctions::Destroy() {
   /* Close the file */
   m_cOutput.close();
// #if defined(USE_ROS_THREAD) || defined(USE_ROS_SAFE_QUEUE)
   // m_ros_thread.join();
// #endif
}

/****************************************/
/****************************************/
CColor CForagingLoopFunctions::GetFloorColor(const CVector2& c_position_on_plane) {
    if((c_position_on_plane - m_cFoodPos).SquareLength() < m_fFoodSquareRadius) {
        return CColor::BLACK;       //Food Area
    }
    if((c_position_on_plane - m_cNestPos).SquareLength() < m_fFoodSquareRadius) {
        return CColor::GRAY50;      //Nest Area
    }
    return CColor::WHITE;
}

std::string getCurTimeString() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

bool CForagingLoopFunctions::IsExperimentFinished() {
    /* Check whether a robot is on a food item */
    /*CSpace::TMapPerType& m_cFootbots = GetSpace().GetEntitiesByType("foot-bot");
    bool bAllFinished = true;
    size_t nTotalFoodReturnToNest = 0;
    for(CSpace::TMapPerType::iterator it = m_cFootbots.begin(); it != m_cFootbots.end(); ++it) {
        // Get handle to foot-bot entity and controller //
        CFootBotEntity& cFootBot = *any_cast<CFootBotEntity*>(it->second);
        CFootBotForaging& cController = dynamic_cast<CFootBotForaging&>(cFootBot.GetControllableEntity().GetController());
        if(cController.GetStateData().loopCounter < 20000)
            bAllFinished = false;
        nTotalFoodReturnToNest += cController.GetStateData().nFoodReturnToNest;
    }
    if(bAllFinished) {
        std::cout << "Total " << nTotalFoodReturnToNest << " dropped off food after " << 20000 << " steps" << std::endl;
        m_cOutput << "[" << getCurTimeString() << "]Total " << nTotalFoodReturnToNest << " dropped off food after " 
                  << 20000 << " steps" << std::endl;
    }
    return bAllFinished;*/
    return false;
}

/****************************************/
/****************************************/
// bool CForagingLoopFunctions::ServiceFunction(const ma_foraging::AIServiceRequest &req,
                                             // ma_foraging::AIServiceResponse &resp) {
// #ifdef USE_ROS_THREAD
    // std::unique_lock<std::mutex> lk(m_m_main);
    // m_req_store = req;
    // m_service_data_available = true;
    // lk.unlock();
    // m_cv_main.notify_one();
    // lk.lock();
    // while (!m_loop_done) {
        //printf("[CForagingLoopFunctions::ServiceFunction] %d wait for m_loop_done\n", m_episode_time);
        // m_cv_main.wait(lk);
    // }
    // resp = m_resp_store;
    // m_loop_done = false;
    // m_resp_store = ma_foraging::AIServiceResponse();
    // lk.unlock();
    // return true;
// #elif USE_ROS_SAFE_QUEUE
    // m_req_queue.push(req);
    // m_resp_queue.wait_and_pop(resp);
    // return true;
// #endif
// }

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if(std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

/****************************************/
/****************************************/
void CForagingLoopFunctions::PreStep() {
    if(!m_bAcceptClient) {
        std::ifstream is("/etc/resolv.conf");
        std::string ip = "127.0.0.1";
        /*if(is.good()) {
            std::stringstream buffer;
            buffer << is.rdbuf();
            std::string str = buffer.str();
            std::size_t sIdx = str.find("nameserver ");
            std::size_t eIdx = str.find("\n", sIdx+1);
            ip = trim(str.substr(sIdx+11, eIdx-sIdx-11));
            printf("[CForagingLoopFunctions::PreStep] IP=[%s]", ip.c_str());
        }*/
        ip = m_sHost;
        m_sClient.Init(ip, m_nPort);
        printf("[CForagingLoopFunctions::PreStep] Client connected to server\n");
        m_bAcceptClient = true;
    }
    
    if(m_persistence_cnt == 0) { 
        //only at first episode time or after certain persistence time 
        // => receive new decision, otherwise keep using previous decision
        //convert input byte array into m_req_store
        //std::string jsonReqStr = m_sClient.Receive();
        //++20201205 handle send recv socket error
        std::string jsonReqStr = "";
        bool bRecv = m_sClient.Receive(jsonReqStr);
        if(!bRecv) {
            m_sClient.Init(m_sHost, m_nPort);
            printf("[CForagingLoopFunctions::PreStep] Reconnect to server\n");
            bool bRecv = m_sClient.Receive(jsonReqStr);
            if(!bRecv)
                exit(-1);
        }
        Json::Value jsonReq;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse( jsonReqStr, jsonReq );
        m_req_store = ma_foraging::AIServiceRequest();
        const Json::Value actValues = jsonReq["actions"];
        for ( int i = 0; i < actValues.size(); i++ ) {
            m_req_store.decisions.push_back(actValues[i].asInt());
        }
        const Json::Value posesValues = jsonReq["start_poses"];
        for ( int i = 0; i < posesValues.size(); i++ ) {
            geometry_msgs::Pose2D pose;
            pose.x = posesValues[i][0].asFloat();
            pose.y = posesValues[i][1].asFloat();
            pose.theta = posesValues[i][2].asFloat();
            m_req_store.reset_poses.push_back(pose);
        }
        // reset or set new action
        auto robots_map = GetSpace().GetEntitiesByType("foot-bot");
        //std::size_t i = 0;
        size_t nTotalFoodReturnToNest = 0;
        for (const auto &elem : robots_map) {
            auto robot = any_cast<CFootBotEntity *>(elem.second);
            CFootBotForaging &cController = dynamic_cast<CFootBotForaging &>(robot->GetControllableEntity().GetController());
            std::size_t robotId = cController.GetStateData().id;
            if (!m_req_store.reset_poses.empty()) {
                assert(m_req_store.reset_poses.size() == robots_map.size());
                const auto &pos = m_req_store.reset_poses[robotId];
                nTotalFoodReturnToNest += cController.GetStateData().nFoodReturnToNest;
                cController.Reset();
                cController.setFoodNestPos(m_cFoodPos, m_cNestPos);
                cController.SetForagingParams(m_sForagingParams);
                auto move_non_colliding = MoveEntity(
                    robot->GetEmbodiedEntity(),     // move the body of the robot
                    convertVec(pos),                // to this position
                    convertQuat(pos),               // with this orientation
                    false);
                //std::cout << "Reset pose: x " << pos.x << " y " << pos.y << " theta: " << pos.theta << "\n";
                if (!move_non_colliding) {
                    //std::cerr << "Resetting position caused collision!" << std::endl;
                }
                //cController.SetNewAction(0);
            } else {
                assert(robots_map.size() == m_req_store.decisions.size());
                cController.SetNewAction(m_req_store.decisions[robotId], true);
                m_accRewards[robotId] = 0.0f;
            }
            //++i;
        }
        if (!m_req_store.reset_poses.empty()) {
            std::cout << "=== Collected " << nTotalFoodReturnToNest << " => Reset ===\n";
            m_episode_time=0;
            m_persistence_cnt = 0;
#ifdef DEBUG_LOG
            printf("[PreStep] %d %d Reset all robots\n", m_episode_time, m_persistence_cnt);
#endif
        } else {
            ++m_episode_time;
            m_persistence_cnt = 0;
#ifdef DEBUG_LOG
            printf("[PreStep] %d %d Set new action for all robots\n", m_episode_time, m_persistence_cnt);
#endif
        }
    } else {
        //keep using previous decision
        auto robots_map = GetSpace().GetEntitiesByType("foot-bot");
        for (const auto &elem : robots_map) {
            auto robot = any_cast<CFootBotEntity *>(elem.second);
            CFootBotForaging &cController = dynamic_cast<CFootBotForaging &>(robot->GetControllableEntity().GetController());
            std::size_t robotId = cController.GetStateData().id;
            cController.SetNewAction(m_req_store.decisions[robotId], false);
        }
        ++m_episode_time;
    }
}

/****************************************/
/****************************************/
void CForagingLoopFunctions::PostStep() {
#ifdef DEBUG_LOG
    printf("======================= [PostStep] %d %d ===================\n", m_episode_time, m_persistence_cnt);
#endif
    auto robots_map = GetSpace().GetEntitiesByType("foot-bot");
    auto counter = 0;
    for (const auto &elem : robots_map) {
        //std::cout << "**** ID: " << counter << " ********\n";
        auto robot = any_cast<CFootBotEntity *>(elem.second);
        CFootBotForaging &cController = dynamic_cast<CFootBotForaging &>(robot->GetControllableEntity().GetController());
        std::size_t robotId = cController.GetStateData().id;
        if(!cController.isFinishUpdateState()) {
            timespec sleepValue = {0};
            sleepValue.tv_nsec = INTERVAL_MS;
            nanosleep(&sleepValue, NULL);
        }
        ma_foraging::Observation ob = cController.getObservation();
        double reward = cController.GetStateData().reward;
        bool done = false;
        //std::cout << "Final reward " << reward << std::endl;
        if(m_persistence_cnt == 0) { 
            if(!m_req_store.reset_poses.empty()) {
                m_resp_store.observations.push_back(ob);
                m_resp_store.dones.push_back(done);
                m_resp_store.rewards.push_back(reward);
                //m_resp_store.ids.push_back(static_cast<unsigned int>(robotId));
                m_mapIndexes[robotId] = counter;
                //++++++ 20200408 add new observation - message table of robots/foot-bot/simulator/footbot_entity
                std::unordered_map<UInt8, CFootBotForaging::SMessage*> msgTbl = cController.GetMessageTable();
                Json::Value arrMsg;
                for( const auto& n : msgTbl ) {
                    Json::Value msgJson;
                    msgJson["id"] = n.second->id;
                    msgJson["fHop"] = n.second->fHop;
                    msgJson["nHop"] = n.second->nHop;
                    msgJson["dist"] = n.second->dist;
                    msgJson["angle"] = n.second->angle;
                    arrMsg.append(msgJson);
                }
                m_mapMsgTbls[robotId] = arrMsg;
                //------ 20200408
#ifdef DEBUG_LOG
                printf("[PostStep] %d %d Robot %zu - reset reward=%f\n", 
                       m_episode_time, m_persistence_cnt, robotId, reward);
#endif
            } else {
                //assert(robotId < m_accRewards.size());
                m_accRewards[robotId] += reward;
                //++++++20200228 +++++++++++
                if(cController.GetStateData().PickUpEvent) {
                    m_accRewards[cController.GetStateData().PickUpBeaconId] += 
                        m_sForagingParams.PICKUP_FOOD_REWARD*m_sForagingParams.RewardSplitRatio;
                    //if(cController.GetStateData().loopCounter < 5000)
                    //    printf("[PostStep] %d %d both %zu and %zu receive half pickup food reward\n", 
                    //        m_episode_time, m_persistence_cnt, robotId, cController.GetStateData().PickUpBeaconId);
                    cController.GetStateData().PickUpEvent = false;
                    cController.GetStateData().PickUpBeaconId = UNDEFINED_TARGET;
                }
                if(cController.GetStateData().ReturnFoodEvent) {
                    m_accRewards[cController.GetStateData().ReturnBeaconId] += 
                        m_sForagingParams.RETURN_FOOD_REWARD*m_sForagingParams.RewardSplitRatio;
                    //if(cController.GetStateData().loopCounter < 5000)
                    //    printf("[PostStep] %d %d both %zu and %zu receive half return food reward\n", 
                    //        m_episode_time, m_persistence_cnt, robotId, cController.GetStateData().ReturnBeaconId);
                    cController.GetStateData().ReturnFoodEvent = false;
                    cController.GetStateData().ReturnBeaconId = UNDEFINED_TARGET;
                }
                //--------------------------
#ifdef DEBUG_LOG
                printf("[PostStep] %d %d Robot %zu - 1st Set reward=%f\n", 
                       m_episode_time, m_persistence_cnt, robotId, m_accRewards[robotId]);
#endif
            }
        } else {
            //assert(robotId < m_accRewards.size());
            m_accRewards[robotId] += reward;
            //++++++20200228 +++++++++++
            if(cController.GetStateData().PickUpEvent) {
                m_accRewards[cController.GetStateData().PickUpBeaconId] += 
                    m_sForagingParams.RETURN_FOOD_REWARD*m_sForagingParams.RewardSplitRatio;
                //if(cController.GetStateData().loopCounter < 5000)
                //    printf("[PostStep] %d %d both %zu and %zu receive half pickup food reward\n", 
                //            m_episode_time, m_persistence_cnt, robotId, cController.GetStateData().PickUpBeaconId);
                cController.GetStateData().PickUpEvent = false;
                cController.GetStateData().PickUpBeaconId = UNDEFINED_TARGET;
            }
            if(cController.GetStateData().ReturnFoodEvent) {
                m_accRewards[cController.GetStateData().ReturnBeaconId] += 
                    m_sForagingParams.RETURN_FOOD_REWARD*m_sForagingParams.RewardSplitRatio;
                //if(cController.GetStateData().loopCounter < 5000)
                //    printf("[PostStep] %d %d both %zu and %zu receive half return food reward\n", 
                //            m_episode_time, m_persistence_cnt, robotId, cController.GetStateData().ReturnBeaconId);
                cController.GetStateData().ReturnFoodEvent = false;
                cController.GetStateData().ReturnBeaconId = UNDEFINED_TARGET;
            }
            //--------------------------
            if(m_persistence_cnt == m_sForagingParams.PERSISTENCE_CNT-1) {
                m_resp_store.observations.push_back(ob);
                m_resp_store.dones.push_back(done);
                m_resp_store.rewards.push_back(m_accRewards[robotId]);
                //m_resp_store.ids.push_back(static_cast<unsigned int>(robotId));
                m_mapIndexes[robotId] = counter;
                //++++++ 20200408 add new observation - message table of robots/foot-bot/simulator/footbot_entity
                std::unordered_map<UInt8, CFootBotForaging::SMessage*> msgTbl = cController.GetMessageTable();
                Json::Value arrMsg;
                for( const auto& n : msgTbl ) {
                    Json::Value msgJson;
                    msgJson["id"] = n.second->id;
                    msgJson["fHop"] = n.second->fHop;
                    msgJson["nHop"] = n.second->nHop;
                    msgJson["dist"] = n.second->dist;
                    msgJson["angle"] = n.second->angle;
                    arrMsg.append(msgJson);
                }
                m_mapMsgTbls[robotId] = arrMsg;
                //------ 20200408
#ifdef DEBUG_LOG
                printf("[PostStep] %d %d Robot %zu - Final push reward=%f\n", 
                        m_episode_time, m_persistence_cnt, robotId, m_accRewards[robotId]);
#endif
            }
        }
        ++counter;
    }
    
    if(!m_req_store.reset_poses.empty() or m_persistence_cnt == m_sForagingParams.PERSISTENCE_CNT-1) { 
        //when reset or when m_persistence_cnt reach max persistence cnt
        //convert m_resp_store to CByteArray to send over socket
        Json::StreamWriterBuilder builder;
        builder.settings_["indentation"] = "";
        builder.settings_["precision"] = 4;
        Json::Value root;
        Json::Value array;
        for ( size_t i = 0; i < m_resp_store.observations.size(); i++ ) {
            ma_foraging::Observation ob = m_resp_store.observations[m_mapIndexes[i]];
            Json::Value obJson;
            obJson["curState"] = ob.curState;
            //obJson["walkerPersistenceCnt"] = ob.walkerPersistenceCnt;
            //obJson["beaconPersistenceCnt"] = ob.beaconPersistenceCnt;
            obJson["numBeacons"] = ob.numBeacons;
            obJson["hasFood"] = ob.hasFood;
            obJson["minDist"] = ob.minDist;
            obJson["maxDist"] = ob.maxDist;
            obJson["foodClosest"] = ob.foodClosest;
            obJson["nestClosest"] = ob.nestClosest;
            obJson["foodFarthest"] = ob.foodFarthest;
            obJson["nestFarthest"] = ob.nestFarthest;
            obJson["numFoodReturn"] = ob.numFoodReturn;
            //+++++++++ 20200228 +++++++++++++++++++++++++++
            obJson["foodDist"] = ob.foodDist;
            obJson["nestDist"] = ob.nestDist;
            obJson["numWalkers"] = ob.numWalkers;
            //obJson["fHop"] = ob.fHop;             //20200303
            //obJson["nHop"] = ob.nHop;             //20200303
            //++++++ 20200408 add new observation - message table of robots/foot-bot/simulator/footbot_entity
            obJson["msgTbl"] = m_mapMsgTbls[i];
            //------ 20200408 add new observation - message table of robots/foot-bot/simulator/footbot_entity
            //----------------------------------------------
            //printf("%s\n", obJson.toStyledString().c_str());
            array.append(obJson);
        }
        root["observations"] = array;
        
        Json::Value arrRewards;
        for ( size_t i = 0; i < m_resp_store.rewards.size(); i++ ) {
            arrRewards.append(m_resp_store.rewards[m_mapIndexes[i]]);
        }
        root["rewards"] = arrRewards;
        //printf("%s\n", arrRewards.toStyledString().c_str());
        
        Json::Value arrDones;
        for ( size_t i = 0; i < m_resp_store.dones.size(); i++ ) {
            arrDones.append(m_resp_store.dones[m_mapIndexes[i]]);
        }
        root["dones"] = arrDones;
        //printf("%s\n", arrDones.toStyledString().c_str());
        
        /*Json::Value arrIds;
        for ( size_t i = 0; i < m_resp_store.ids.size(); i++ ) {
            arrIds.append(m_resp_store.ids[i]);
        }
        root["ids"] = arrIds;
        //printf("%s\n", arrIds.toStyledString().c_str());*/
        //++++ 20200420 add code to send the foraging params to python to log, only the first time
        if(m_episode_time <= 1) {
            Json::Value obForagingParams;
            obForagingParams["m_fFoodSquareRadius"] = m_sForagingParams.m_fFoodSquareRadius;
            obForagingParams["m_fFoodPosX"] = m_sForagingParams.m_fFoodPosX;
            obForagingParams["m_fFoodPosY"] = m_sForagingParams.m_fFoodPosY;
            obForagingParams["m_fNestPosX"] = m_sForagingParams.m_fNestPosX;
            obForagingParams["m_fNestPosY"] = m_sForagingParams.m_fNestPosY;
            obForagingParams["STAY_REWARD"] = m_sForagingParams.STAY_REWARD;
            obForagingParams["SWITCH_REWARD"] = m_sForagingParams.SWITCH_REWARD;
            obForagingParams["PICKUP_FOOD_REWARD"] = m_sForagingParams.PICKUP_FOOD_REWARD;
            obForagingParams["RETURN_FOOD_REWARD"] = m_sForagingParams.RETURN_FOOD_REWARD;
            obForagingParams["BEACON_REWARD"] = m_sForagingParams.BEACON_REWARD;
            obForagingParams["CROWDED_REWARD"] = m_sForagingParams.CROWDED_REWARD;
            obForagingParams["SPARSE_REWARD"] = m_sForagingParams.SPARSE_REWARD;
            obForagingParams["INF_CARDINALITY_REWARD"] = m_sForagingParams.INF_CARDINALITY_REWARD;
            obForagingParams["ZERO_CARDINALITY_REWARD"] = m_sForagingParams.ZERO_CARDINALITY_REWARD;
            obForagingParams["ZERO_CARDINALITY_METHOD"] = m_sForagingParams.ZERO_CARDINALITY_METHOD;
            obForagingParams["PERSISTENCE_CNT"] = m_sForagingParams.PERSISTENCE_CNT;
            obForagingParams["PROBABILITY"] = m_sForagingParams.PROBABILITY;
            obForagingParams["MaxBeaconDistance"] = m_sForagingParams.MaxBeaconDistance;
            obForagingParams["MaxDistanceDetectFoodNest"] = m_sForagingParams.MaxDistanceDetectFoodNest;
            obForagingParams["TurnToTargetRate"] = m_sForagingParams.TurnToTargetRate;
            obForagingParams["RewardSplitRatio"] = m_sForagingParams.RewardSplitRatio;
            root["params"] = obForagingParams;
        }
        //---- 20200420 
        
        //send to python server
        std::string respStr = Json::writeString(builder, root);
        if(m_episode_time <= 1)
            printf("[PostStep] %d - Start Send %d bytes\n", m_episode_time, respStr.length());
        bool bSend = m_sClient.Send(respStr);
        if(!bSend) {
            m_sClient.Init(m_sHost, m_nPort);
            printf("[CForagingLoopFunctions::PostStep] Reconnect to server\n");
            bool bSend = m_sClient.Send(respStr);
            if(!bSend)
                exit(-1);
        }
        m_resp_store = ma_foraging::AIServiceResponse();
        if(m_episode_time <= 1)
            printf("[PostStep] Finish Send %d bytes\n", respStr.length());
        if(m_persistence_cnt == m_sForagingParams.PERSISTENCE_CNT-1) {
#ifdef DEBUG_LOG
            printf("[PostStep] %d %d END, now reset persistence cnt to 0\n", m_episode_time, m_persistence_cnt);
#endif
            m_persistence_cnt = 0;
        }
    } else {
        m_persistence_cnt += 1;
        if(m_persistence_cnt == m_sForagingParams.PERSISTENCE_CNT)
            m_persistence_cnt = 0;
        //printf("[PostStep] %d %d END\n", m_episode_time, m_persistence_cnt);
    }
}

/****************************************/
/****************************************/
REGISTER_LOOP_FUNCTIONS(CForagingLoopFunctions, "foraging_loop_functions")
