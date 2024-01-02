#include <iostream>
#include <vector>
#include "assert.h"
using namespace std;

struct ans{
    int row1_signal_number;
    int row2_signal_number;
    double coor_difference;
};

ans WSproblem(int n,int m, double W1,double S1,double W2,double S2)//row1: n, W1, S1, row2: m, W2, W2
{
    assert(n%2==0);
    assert(m%2==0);
    //S G G | G G S G G G G S, consider only the right side, use lower left coordinate for wires, and set center of the layer as origin

    int row1_pointer=0;
    int row2_pointer=0;

    vector<double> row1_coor;
    vector<double> row2_coor;

    row1_coor.push_back(2.0*W1+2.0*S1+S1/2.0);
    row2_coor.push_back(2.0*W2+2.0*S2+S2/2.0);

    ans best;
    best.row1_signal_number=0;
    best.row2_signal_number=0;
    best.coor_difference=abs(row1_coor[0]-row2_coor[0]);

    while(row1_pointer<n/2-1&&row2_pointer<m/2-1)
    {
        if (row1_coor[row1_pointer]<row2_coor[row2_pointer])
        {
            row1_coor.push_back(row1_coor[row1_pointer]+5.0*W1+5.0*S1);
            row1_pointer++;
        }
        else if((row1_coor[row1_pointer]>row2_coor[row2_pointer]))
        {
            row2_coor.push_back(row2_coor[row2_pointer]+5.0*W2+5.0*S2);
            row2_pointer++;
        }
        
        double temp=abs(row1_coor[row1_pointer]-row2_coor[row2_pointer]);
        
        if (temp<best.coor_difference)
        {
            best.row1_signal_number=row1_pointer;
            best.row2_signal_number=row2_pointer;
            best.coor_difference=temp;
        }
        if(temp==0)
        {
            break;
        }        
    }

    while(row1_pointer<n/2-1)
    {
        if (row1_coor[row1_pointer]>row2_coor[row2_pointer])
        {
            break;// will never catch up again
        }
    
        row1_coor.push_back(row1_coor[row1_pointer]+5.0*W1+5.0*S1);
        row1_pointer++;
    
        double temp=abs(row1_coor[row1_pointer]-row2_coor[row2_pointer]);

        if (temp<best.coor_difference)
        {
            best.row1_signal_number=row1_pointer;
            best.row2_signal_number=row2_pointer;
            best.coor_difference=temp;
        }
        if(temp==0)
        {
            break;
        }        
    }

    while (row2_pointer<m/2-1)
    {
        if (row1_coor[row1_pointer]<row2_coor[row2_pointer])
        {
            break;// will never catch up again
        }
    
        row2_coor.push_back(row2_coor[row2_pointer]+5.0*W2+5.0*S2);
        row2_pointer++;
    
        double temp=abs(row1_coor[row1_pointer]-row2_coor[row2_pointer]);
        
        if (temp<best.coor_difference)
        {
            best.row1_signal_number=row1_pointer;
            best.row2_signal_number=row2_pointer;
            best.coor_difference=temp;
        }
        if(temp==0)
        {
            break;
        }        
    }
    

    cout<<best.row1_signal_number<<" "<<best.row2_signal_number<<" "<<best.coor_difference<<endl;
    cout<<row1_coor[best.row1_signal_number]<<" "<<row2_coor[best.row2_signal_number]<<endl;//! start from 0!
    
    cout<<"row 1 coors: ";
    for(double row1:row1_coor)
    {
        cout<<row1<<" ";
    }
    cout<<"\nrow 2 coors: ";
    for(double row2:row2_coor)
    {
        cout<<row2<<" ";
    }
    cout<<endl;
    
    return best;

}

int main(int argc, char * argv[])
{
    while(1)
    {
        if(argc!=7)
        {
            cout<<"Please enter: n,m,w1,s1,w2,s2\n";
        }
        else{
            ans best=WSproblem(atoi(argv[1]),atoi(argv[2]),strtod(argv[3],NULL),strtod(argv[4],NULL),strtod(argv[5],NULL),strtod(argv[6],NULL));
            break;
        }
    }


}