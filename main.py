from model_architecture import ObjectDetectionModel
from preprocessing import ModelRunners
path = 'Model/PeopleDetectionModel.pth'

if __name__ == "__main__" :
    model_runners = ModelRunners(path)

    while True :
        print("############### Main ###############")
        print("1. run system")
        print("2. set video size ")
        print("3. set threshold size")
        print("4. set web_cam (1 for external) (0 for internal)")
        print("0. quit")
        choice = int(input("choice option "))

        if choice == 0 :
            break 
        elif choice == 1 :
            model_runners.main_model()
        elif choice == 2 :
            weidth = int(input("set weidth  : "))
            heigth = int(input("set heigth : " ))
            model_runners.set_imagesize((weidth,heigth))
        elif choice == 3 :
            model_runners.set_threshold()
        elif choice == 4 :
            web_cam = int(input("choice (0 for internal camera) (1 for external camera)"))
            if web_cam > 1 :
                print("Warning : choice just run for 1 / 0 input set to default 0")
                model_runners.set_webcam(0)
            else :
                model_runners.set_webcam(1)
        else :
            print("option is not aivable")
        
