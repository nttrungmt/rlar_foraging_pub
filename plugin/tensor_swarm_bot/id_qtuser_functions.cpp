#include "id_qtuser_functions.h"
#include "footbot_foraging.h"
#include <sstream>              // std::stringstream
 
/****************************************/
/****************************************/
 
CIDQTUserFunctions::CIDQTUserFunctions() {
   RegisterUserFunction<CIDQTUserFunctions,CFootBotEntity>(&CIDQTUserFunctions::Draw);
}
 
/****************************************/
/****************************************/
 
void CIDQTUserFunctions::Draw(CFootBotEntity& c_entity) {
   	/* The position of the text is expressed wrt the reference point of the footbot
     * For a foot-bot, the reference point is the center of its base.
     * See also the description in
     * $ argos3 -q foot-bot
     */
   	//DrawText(CVector3(0.0, 0.0, 0.3),   // position
   	//         c_entity.GetId().c_str()); // text
   	CFootBotForaging& cController = dynamic_cast<CFootBotForaging&>(c_entity.GetControllableEntity().GetController());
	std::string strAction = "";
	switch(cController.GetStateData().Action) {
		case CFootBotForaging::SStateData::ACTION_AVOID:
			strAction = "Av";
			break;
		case CFootBotForaging::SStateData::ACTION_EXPLORE:
			strAction = "Ex";
			break;
		case CFootBotForaging::SStateData::ACTION_PURSUE:
			strAction = "Ps";
			break;
		default:
			break;
	}
	std::string strMove = "";
	switch(cController.GetStateData().Movement) {
		case CFootBotForaging::SStateData::MOVE_STATIONARY:
			strMove = "St";
			break;
		case CFootBotForaging::SStateData::MOVE_FORWARD:
			strMove = "Fw";
			break;
		case CFootBotForaging::SStateData::MOVE_TURN:
			strMove = "Tu";
			break;
		default:
			break;
	}
   	std::stringstream ss;
   	ss << cController.GetId()
      << ',' << cController.GetStateData().myFoodHop 
      << ',' << cController.GetStateData().myNestHop
      << "," << cController.GetStateData().target_id 
      << "," << strAction
      << "," << strMove;
   	DrawText(CVector3(0.1, 0.0, 0.3),   // position
            ss.str().c_str());         // text
}
 
/****************************************/
/****************************************/
 
REGISTER_QTOPENGL_USER_FUNCTIONS(CIDQTUserFunctions, "id_qtuser_functions")
