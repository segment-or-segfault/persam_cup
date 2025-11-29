import { useState } from "react";
import { ImageUpload } from "@/components/ImageUpload";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Sparkles, ArrowRight } from "lucide-react";
import { toast } from "sonner";

const Index = () => {
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [referenceImageMask, setReferenceImageMask] = useState<File | null>(null);
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultMask, setResultMask] = useState<string | null>(null);

  const handleProcess = async () => {
    if (!referenceImage || !referenceImageMask || !queryImage) {
      toast.error("Please upload all three images");
      return;
    }

    setIsProcessing(true);
    setResultMask(null);
    try {
      const form = new FormData();
      form.append("ref_image", referenceImage);
      form.append("ref_mask", referenceImageMask);
      form.append("test_image", queryImage);

      const res = await fetch("http://localhost:3000/segment", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        throw new Error("Backend processing failed");
      }

      // backend returns PNG binary → turn into object URL
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResultMask(url);

      toast.success("Mask generated successfully!");
    } catch (err) {
      console.error(err);
      toast.error("Failed to generate mask");
    }

    setIsProcessing(false);
  };

  const canProcess = referenceImage && referenceImageMask && queryImage && !isProcessing;

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Hero Section */}
      <header className="border-b border-border/50 bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-2">
            <div className="rounded-lg bg-gradient-primary p-2">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-foreground">PerSAM</h1>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        {/* Title and Description */}
        <div className="mb-12 text-center">
          <h2 className="mb-4 text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
            Personalized Segment Anything
          </h2>
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
            Upload a reference image and a query image to generate precise segmentation masks.
            Our model learns from your reference to identify similar objects in the query image.
          </p>
        </div>

        {/* Main Content */}
        <div className="mx-auto max-w-5xl">
          <Card className="overflow-hidden border-border bg-card shadow-medium">
            <div className="p-8">
              {/* Upload Section */}
              <div className="mb-8 grid gap-8 md:grid-cols-3">
                <ImageUpload
                  label="Reference Image"
                  description="Upload the image containing the object you want to segment"
                  onImageSelect={setReferenceImage}
                  image={referenceImage}
                />
                <ImageUpload
                  label="Reference Image Mask"
                  description="Upload the image mask corresponding to the reference image"
                  onImageSelect={setReferenceImageMask}
                  image={referenceImageMask}
                />
                <ImageUpload
                  label="Query Image"
                  description="Upload the image where you want to find similar objects"
                  onImageSelect={setQueryImage}
                  image={queryImage}
                />
              </div>

              {/* Process Button */}
              <div className="flex justify-center">
                <Button
                  onClick={handleProcess}
                  disabled={!canProcess}
                  size="lg"
                  className="group relative overflow-hidden bg-gradient-primary px-8 text-base font-semibold shadow-soft transition-all hover:shadow-medium disabled:opacity-50"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    {isProcessing ? (
                      <>
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        Processing...
                      </>
                    ) : (
                      <>
                        Generate Mask
                        <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                      </>
                    )}
                  </span>
                </Button>
              </div>

              {/* Result Section */}
              {resultMask && (
                <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="mb-3 text-center">
                    <h3 className="text-lg font-semibold text-foreground">Generated Mask</h3>
                    <p className="text-sm text-muted-foreground">Segmentation result</p>
                  </div>
                  <div className="flex justify-center">
                    <div className="relative overflow-hidden rounded-lg border border-border bg-muted shadow-medium">
                      <img 
                        src={resultMask} 
                        alt="Generated mask result" 
                        className="max-h-96 w-auto object-contain"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Info Cards */}
          <div className="mt-12 grid gap-6 md:grid-cols-3">
            <Card className="border-border bg-card p-6 shadow-soft">
              <div className="mb-3 inline-flex rounded-lg bg-primary/10 p-2">
                <span className="text-2xl">🎯</span>
              </div>
              <h3 className="mb-2 font-semibold text-foreground">
                Precise Segmentation
              </h3>
              <p className="text-sm text-muted-foreground">
                Advanced deep learning for accurate object identification
              </p>
            </Card>

            <Card className="border-border bg-card p-6 shadow-soft">
              <div className="mb-3 inline-flex rounded-lg bg-accent/10 p-2">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="mb-2 font-semibold text-foreground">
                Fast Processing
              </h3>
              <p className="text-sm text-muted-foreground">
                Get results in seconds with optimized inference
              </p>
            </Card>

            <Card className="border-border bg-card p-6 shadow-soft">
              <div className="mb-3 inline-flex rounded-lg bg-primary/10 p-2">
                <span className="text-2xl">🔄</span>
              </div>
              <h3 className="mb-2 font-semibold text-foreground">
                Adaptive Learning
              </h3>
              <p className="text-sm text-muted-foreground">
                Model adapts to your reference for better accuracy
              </p>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-24 border-t border-border/50 bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-8 text-center text-sm text-muted-foreground">
          <p>Powered by DINOv2 and Segment Anything Model</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
