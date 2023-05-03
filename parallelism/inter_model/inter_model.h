////
//// Created by Anastasiaa on 05.04.2023.
////
//
//#ifndef DEEPDENDRO_INTER_MODEL_H
//#define DEEPDENDRO_INTER_MODEL_H
//
//#include "ts_queue.h"
//#include "Model.h"
//
//struct ModelObject {
//    Model obj;
//    size_t epoch;
//    double learning_rate;
//    bool isEmpty() {return (epoch==0);};
//};
//
//class InterModel {
//    TSQueue<ModelObject> ts_queue;
//    std::vector<std::thread> threads;
//
//public:
//    InterModel () = default;
//
//    void addModel (const Model& mdl, size_t epochs, double learning_rate);
//
//    void runThreads (int num_threads);
//
//    static void trainModels (TSQueue<ModelObject> &ts_queue);
//};
//
//
//#endif //DEEPDENDRO_INTER_MODEL_H
