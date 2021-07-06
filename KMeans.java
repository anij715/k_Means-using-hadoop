import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Vector;
import java.text.DecimalFormat;
import java.util.ArrayList;
 
import org.apache.hadoop.conf.*;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.*;
 
public class KMeans extends Configured implements Tool
{
    private static final int MAXITERATIONS = 2;
    private static final double THRESHOLD = 30;
    private static boolean StopSignalFromReducer=false;
    private static int NoChangeCount=0;
    public static class Point implements Writable
    {
		public static final int DIMENTION=11;
		public double arr[];

		public Point()
		{
			arr=new double[DIMENTION];
			for(int i=0;i<DIMENTION;++i)
			{
				arr[i]=0;
			}
		}

		 
		public static double getEuclidDist(Point vec1,Point vec2)
		{
			if(!(vec1.arr.length==DIMENTION && vec2.arr.length==DIMENTION))
			{
				System.exit(1);
			}
			double dist=0.0;
			for(int i=0;i<DIMENTION;++i)
			{
				dist+=(vec1.arr[i]-vec2.arr[i])*(vec1.arr[i]-vec2.arr[i]);
			}
			return Math.sqrt(dist);
		}

		 
		public void clear()
		{
			for(int i=0;i<arr.length;i++)
			{
				arr[i]=0.0;
			}
		}
		 
		public String toString()
		{
			DecimalFormat df=new DecimalFormat("0.0000");
			String rect=String.valueOf(df.format(arr[0]));
			for(int i=1;i<DIMENTION;i++)
			{
				rect+=","+String.valueOf(df.format(arr[i]));
			}
			return rect;
		}
	 
		@Override
		public void readFields(DataInput in) throws IOException 
		{
			String str[]=in.readUTF().split(",");
			for(int i=0;i<DIMENTION;++i)
			{
				arr[i]=Double.parseDouble(str[i]);
			}
		}
	 
		@Override
		public void write(DataOutput out) throws IOException 
		{
			out.writeUTF(this.toString());
		}
	}
	
    public static boolean stopIteration(Configuration conf) throws IOException //called in main
    {
        FileSystem fs=FileSystem.get(conf);
        Path pervCenterFile=new Path("your path");
        Path currentCenterFile=new Path("your path");
        if(!(fs.exists(pervCenterFile) && fs.exists(currentCenterFile)))
        {
            System.exit(1);
        }
        //check whether the centers have changed or not to determine to do iteration or not
        boolean stop=true;
        String line1,line2;
        FSDataInputStream in1=fs.open(pervCenterFile);
        FSDataInputStream in2=fs.open(currentCenterFile);
        InputStreamReader isr1=new InputStreamReader(in1);
        InputStreamReader isr2=new InputStreamReader(in2);
        BufferedReader br1=new BufferedReader(isr1);
        BufferedReader br2=new BufferedReader(isr2);
        Point prevCenter,currCenter;
        while((line1=br1.readLine())!=null && (line2=br2.readLine())!=null)
        {
            prevCenter=new Point();
            currCenter=new Point();
            String []str1=line1.split(",");
            String []str2=line2.split(",");
            for(int i=0;i<Point.DIMENTION;i++)
            {
                prevCenter.arr[i]=Double.parseDouble(str1[i]);
                currCenter.arr[i]=Double.parseDouble(str2[i]);
            }
            if(Point.getEuclidDist(prevCenter, currCenter)>THRESHOLD)
            {
                stop=false;
                break;
            }
        }
        //if another iteration is needed, then replace previous controids with current centroids
        if(stop==false)
        {
            fs.delete(pervCenterFile,true);
            if(fs.rename(currentCenterFile, pervCenterFile)==false)
            {
                System.exit(1);
            }
        }
        return stop;
    }
     
    public static class ClusterMapper extends Mapper<LongWritable, Text, Text, Text>  //output<centroid,point>
    {
        Vector<Point> centers = new Vector<Point>();
        Point point=new Point();
        int k=0;
        @Override
        //clear centers
        public void setup(Context context)
        {
            try
            {
				Path[] caches=DistributedCache.getLocalCacheFiles(context.getConfiguration());
				if(caches==null || caches.length<=0)
				{
					System.exit(1);
				}
				
				BufferedReader br=new BufferedReader(new FileReader(caches[0].toString()));
				Point point;
				String line;
				while((line=br.readLine())!=null)
				{
					point=new Point();
					String[] str=line.split(",");
					for(int i=0;i<Point.DIMENTION;i++)
					{
						point.arr[i]=Double.parseDouble(str[i]);
					}
					centers.add(point);
					k++;           
				}
            }
			catch(Exception e){}
        }
        @Override
        //output<centroid,point>
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException 
        {
			int index=-1;
			double s=0.0;
			ArrayList<Double> sse = new ArrayList<>();

			String outputValue;
			double minDist=Double.MAX_VALUE;
            String[] str=value.toString().split(",");
            for(int i=0;i<Point.DIMENTION;i++)
            {
                point.arr[i]=Double.parseDouble(str[i]);
			}

            for(int i=0;i<k;i++)
            {
                double dist=Point.getEuclidDist(point, centers.get(i));
                if(dist<minDist)
                {
                    minDist=dist;
                    index=i;
                    s = minDist*minDist;
                }
                sse.add(s);
                
            }

            outputValue=point.toString()+"-"+sse.get(index).toString();
            context.write(new Text(centers.get(index).toString()), new Text(outputValue));
        }
        
        @Override
        public void cleanup(Context context) throws IOException,InterruptedException 
        {

        }
    }
     
    public static class Combiner extends Reducer<Text, Text, Text, Text> //value=Point_Sum+count
    {      
        @Override
        //update every centroid except the last one
        public void reduce(Text key,Iterable<Text> values,Context context) throws IOException,InterruptedException
        {
			Point sumPoint=new Point();
			double sse1 = 0.0;
			String outputValue;
			int count=0;
            while(values.iterator().hasNext())
            {
				String line=values.iterator().next().toString();

				String[] str2=line.split("-");
				sse1=Double.parseDouble(str2[1]);

                String[] str1=str2[0].split(":");
                
                if(str1.length==2)
                {
					count+=Integer.parseInt(str1[1]);
				}
				
                String[] str=str1[0].split(",");
                for(int i=0;i<Point.DIMENTION;i++)
                {
                    sumPoint.arr[i]+=Double.parseDouble(str[i]);
				}
                count++;


            }
			outputValue=sumPoint.toString()+":"+String.valueOf(count)+"-"+String.valueOf(sse1);
            context.write(key, new Text(outputValue));
        }
    }
 
	public static class UpdateCenterReducer extends Reducer<Text, Text, Text, Text> 
    {
		@Override
        public void setup(Context context)
        {

        }
 
        @Override
        public void reduce(Text key,Iterable<Text> values,Context context) throws IOException,InterruptedException
        {
			int count=0;
			double sse1=0.0;
			String outputValue;
			Point sumPoint=new Point();
			Point newCenterPoint=new Point();
			String outputKey;
            while(values.iterator().hasNext())
            {
                String line=values.iterator().next().toString();
                String[] str2=line.split("-");
				sse1=Double.parseDouble(str2[1]);
                String[] str=str2[0].split(":");
                String[] pointStr=str[0].split(",");
                count+=Integer.parseInt(str[1]);
                for(int i=0;i<Point.DIMENTION;i++)
                {
                    sumPoint.arr[i]+=Double.parseDouble(pointStr[i]);
                    
				}
            }
            for(int i=0;i<Point.DIMENTION;i++)
            {
                newCenterPoint.arr[i] = sumPoint.arr[i]/count;
			}

			String[] str=key.toString().split(",");
			if(newCenterPoint.arr[0]-Double.parseDouble(str[0])<=THRESHOLD && newCenterPoint.arr[1]-Double.parseDouble(str[1])<=THRESHOLD) // compare old and new centroids
            {
                NoChangeCount++;
			}		
			if(NoChangeCount==10)
			{
				StopSignalFromReducer=true;
			}


			outputValue=String.valueOf(sse1)+" Count:"+String.valueOf(count);
			context.write(new Text(newCenterPoint.toString()),new Text("Sum of Squared Errors:"+String.valueOf(outputValue)));
        }
        
        @Override
        public void cleanup(Context context) throws IOException,InterruptedException 
        {

        }


    }
    @Override
    public int run(String[] args) throws Exception 
    {
        Configuration conf=getConf();
        FileSystem fs=FileSystem.get(conf);
        Job job=new Job(conf);
        job.setJarByClass(KMeans.class);
        
        FileInputFormat.setInputPaths(job, "your path");
        Path outDir=new Path("your path");
        fs.delete(outDir,true);
        FileOutputFormat.setOutputPath(job, outDir);
         
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapperClass(ClusterMapper.class);
        job.setCombinerClass(Combiner.class);
        job.setReducerClass(UpdateCenterReducer.class);
        job.setNumReduceTasks(1);//
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
         
        return job.waitForCompletion(true)?0:1;
    }
    public static void main(String[] args) throws Exception 
    {
        Configuration conf = new Configuration();
        FileSystem fs=FileSystem.get(conf);
         
        Path dataFile=new Path("your path");
        DistributedCache.addCacheFile(dataFile.toUri(), conf);
 
        int iteration = 1;
        int success = 1;
        do 
        {
            success ^= ToolRunner.run(conf, new KMeans(), args);
            iteration++;
        } while (success == 1 && iteration <= MAXITERATIONS && (!stopIteration(conf)) && !StopSignalFromReducer);
		
        Job job=new Job(conf);
        job.setJarByClass(KMeans.class);
        
        FileInputFormat.setInputPaths(job, "your path");
        Path outDir=new Path("your path");
        fs.delete(outDir,true);
        FileOutputFormat.setOutputPath(job, outDir);
         
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapperClass(ClusterMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(Point.class);
        job.setOutputValueClass(Point.class);
         
        job.waitForCompletion(true);
        
    }
}
