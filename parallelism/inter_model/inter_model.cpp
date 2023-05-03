////
//// Created by Anastasiaa on 05.04.2023.
////
//
//#include "inter_model.h"
//
//
//void InterModel::addModel(const Model& mdl, size_t epochs, double learning_rate) {
//    ts_queue.push(ModelObject {mdl, epochs, learning_rate});
//}
//
//void InterModel::runThreads(int num_threads) {
//    ts_queue.push(ModelObject {}); // poison pill
//
//    for(int i = 0; i < num_threads; ++i){
//        threads.emplace_back(trainModels, std::ref(ts_queue));
//    }
//
//    for(int i = 0; i < num_threads; ++i){
//        threads[i].join();
//    }
//}
//
//void InterModel::trainModels(TSQueue<ModelObject> &ts_queue) {
//    while (true) {
//        ModelObject mdl = ts_queue.pop();
//
//        if (mdl.isEmpty()) {
//            ts_queue.push(mdl);
//            return;
//        }
//
//        mdl.obj.train(mdl.epoch, mdl.learning_rate);
//    }
//}
//
